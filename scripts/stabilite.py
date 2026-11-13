#!/usr/bin/env python3
"""Test de stabilité : que reste-t-il d'une carte d'encre quand on repasse le modèle autrement ?

Un vrai trait d'encre est retrouvé quel que soit le jeu de poids et quelle que soit la fenêtre de
profondeur donnée au modèle ; un faux positif change d'une passe à l'autre. On relit donc toutes les passes
de `ink9.py` sur une même zone (data/ink9/NOM_d<décalage>.npy, chacune de forme (jeux de poids, H, W),
probabilité x 255) et l'on garde, pixel par pixel, le MINIMUM : ce qui est « encre » dans toutes les passes.

Sorties (affichées) : part de la surface stable, nombre et taille des taches stables, accord entre les
deux jeux de poids (corrélation de leurs cartes à la profondeur 0). Avec --image, une planche
moyenne | minimum est écrite.

  .venv/bin/python -I scripts/stabilite.py p4_texte                          # texte connu : la référence
  .venv/bin/python -I scripts/stabilite.py p4_droite --extrait p4_vierge 30 64 --image apercus/stabilite_droite.png
        # la passe à la profondeur 0 est découpée dans une passe plus large (décalage y, x en mm)
  .venv/bin/python -I scripts/stabilite.py p4_vierge --fenetre 48 106 4 64   # une seule passe : simple part de surface « encre »
"""
import argparse
import glob
import pathlib
import re
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

PX_MM = 0.0096                    # un pixel des cartes de ink9.py = 9,6 µm


def passes(nom, extrait=None):
    """Liste de (étiquette, tableau (jeux, H, W)) pour toutes les passes de la zone."""
    out = []
    for f in sorted(glob.glob(str(V.DATA / "ink9" / f"{nom}_d[+-]*.npy"))):
        out.append((re.search(r"_d([+-]\d+)\.npy$", f).group(1), np.load(f, mmap_mode="r")))
    if not out:
        raise SystemExit(f"aucune passe data/ink9/{nom}_d*.npy")
    if extrait:
        autre, y_mm, x_mm = extrait
        H, W = out[0][1].shape[1:]
        y0, x0 = round(float(y_mm) / PX_MM), round(float(x_mm) / PX_MM)
        for f in sorted(glob.glob(str(V.DATA / "ink9" / f"{autre}_d[+-]*.npy"))):
            a = np.load(f, mmap_mode="r")
            out.append((re.search(r"_d([+-]\d+)\.npy$", f).group(1) + f" ({autre})", a[:, y0:y0 + H, x0:x0 + W]))
    return out


def analyser(nom, extrait=None, seuil=0.5, image=None, fenetre=None):
    P = passes(nom, extrait)
    coupe = (slice(None), slice(None))
    if fenetre:                                                                   # mm, comptés depuis le coin haut gauche de la zone
        coupe = (slice(round(fenetre[0] / PX_MM), round(fenetre[1] / PX_MM)), slice(round(fenetre[2] / PX_MM), round(fenetre[3] / PX_MM)))
        P = [(e, a[(slice(None),) + coupe]) for e, a in P]
    pile = np.concatenate([np.asarray(a) for _, a in P])                         # (passes x jeux, H, W)
    s = int(255 * seuil)                                                         # 0,5 -> 127 : « encre » = valeur >= 128
    mini, moy = pile.min(0), pile.mean(0)
    surface = V.DATA / "ink9" / f"{nom}_surface.npy"
    valide = np.asarray(np.load(surface, mmap_mode="r")[coupe]) > 0 if surface.exists() else np.ones(mini.shape, bool)
    stable = mini > s
    taches, n = ndi.label(stable)
    aires = np.bincount(taches.ravel())[1:] * PX_MM ** 2 if n else np.zeros(1)
    print(f"{nom} : {len(pile)} cartes ({', '.join(e for e, _ in P)}), {mini.shape[0] * PX_MM:.0f} x {mini.shape[1] * PX_MM:.0f} mm, "
          f"{100 * valide.mean():.0f} % de surface couverte par le segment")
    print(f"   encre en moyenne (probabilité moyenne > {seuil}) : {100 * (moy > s)[valide].mean():.2f} % de la surface")
    print(f"   encre STABLE (probabilité > {seuil} dans toutes les cartes) : {100 * stable[valide].mean():.3f} % de la surface ; "
          f"{n} taches, la plus grande {aires.max():.2f} mm²")
    for e, a in P:
        if a.shape[0] >= 2:
            u, v = np.asarray(a[0], np.float32)[valide][::7], np.asarray(a[1], np.float32)[valide][::7]
            print(f"   accord entre les deux premiers jeux de poids, passe {e} : corrélation {np.corrcoef(u, v)[0, 1]:.2f}")
    if image:
        from PIL import Image
        f = max(1, mini.shape[1] // 1600)
        planche = np.concatenate([moy[::f, ::f], np.full((moy[::f, ::f].shape[0], 8), 60.0), mini[::f, ::f].astype(np.float32)], 1)
        pathlib.Path(image).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(planche * 1.6, 0, 255).astype(np.uint8)).save(image)
        print(f"   planche moyenne | minimum : {image}")
    return stable


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nom", help="nom de la zone passée à ink9.py (fichiers data/ink9/NOM_d*.npy)")
    ap.add_argument("--extrait", nargs=3, metavar=("AUTRE", "Y_MM", "X_MM"), help="ajouter les passes d'une zone plus large, découpées à ce décalage")
    ap.add_argument("--seuil", type=float, default=0.5, help="probabilité au-dessus de laquelle un pixel compte comme encre")
    ap.add_argument("--image", help="chemin d'une planche PNG moyenne | minimum")
    ap.add_argument("--fenetre", nargs=4, type=float, metavar=("Y0", "Y1", "X0", "X1"), help="ne garder qu'une fenêtre, en mm depuis le coin haut gauche de la zone")
    a = ap.parse_args()
    analyser(a.nom, a.extrait, a.seuil, a.image, a.fenetre)
