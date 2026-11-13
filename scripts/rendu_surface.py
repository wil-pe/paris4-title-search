#!/usr/bin/env python3
"""Projette un volume 3D (scan ou prédiction d'encre) sur un morceau de segment.

Pour chaque pixel de sortie, on prend la position 3D du maillage (tifxyz), on se déplace
le long de la normale de -T à +T voxels et on garde le maximum rencontré. C'est l'équivalent
d'un « rendu de surface » limité à une petite fenêtre, calculé sans télécharger tout le volume.

Exemple :
  .venv/bin/python -I scripts/rendu_surface.py \\
      --volume PHercParis4/representations/predictions/ink-3d/....zarr/ --niveau 0 --facteur 2 \\
      --maillage data/PHercParis4/segments/<seg>/mesh/<id>-on-20260411134726-2.4um.tifxyz \\
      --fenetre 13000 26000 86000 97000 --pas 2 --demi-epaisseur 24 --sortie apercus/rendu.png
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402


_CACHE_MAILLAGE = {}


def _maillage(dossier):
    """Charge une fois les trois images x, y, z d'un tifxyz et son échelle."""
    dossier = pathlib.Path(dossier)
    if dossier not in _CACHE_MAILLAGE:
        ech = json.loads((dossier / "meta.json").read_text())["scale"][0]   # 0.05 => 1 noeud tous les 20 px
        _CACHE_MAILLAGE[dossier] = ([tifffile.imread(dossier / f) for f in ("x.tif", "y.tif", "z.tif")], 1 / ech)
    return _CACHE_MAILLAGE[dossier]


def rendre(chemin_volume, niveau, facteur, maillage, fenetre, pas=2, T=24, bande=800, stat="max",
           base=V.BUCKET, tranches=None, cache=True):
    """fenetre = (y0, y1, x0, x1) en pixels pleine résolution du segment (grille tifxyz / échelle).
    facteur = taille d'un voxel du niveau lu, en voxels du volume d'origine (1, 2, 4...).
    Le déplacement t le long de la normale est compté en voxels du niveau lu ; t > 0 va vers
    l'extérieur du rouleau (vérifié sur PHerc. Paris 4 : l'encre du recto est vers t < 0).
    Renvoie (image, profondeur) : la statistique le long de la normale et l'indice du maximum.
    Si tranches = [(t0, t1), ...] est donné, renvoie en plus la liste des maxima par tranche."""
    G3, k = _maillage(maillage)
    y0, y1, x0, x1 = fenetre
    r0, r1 = max(0, int(y0 // k) - 2), int(y1 // k) + 3
    c0, c1 = max(0, int(x0 // k) - 2), int(x1 // k) + 3
    G = [g[r0:r1, c0:c1].astype(np.float64) for g in G3]
    invalide = (G[2] <= 0) | (G[0] <= 0)
    P = np.stack(G, -1)                                   # (r, c, 3) en x, y, z
    du = np.gradient(P, axis=1); dv = np.gradient(P, axis=0)
    N = np.cross(du, dv); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    vs = np.arange(y0, y1, pas); us = np.arange(x0, x1, pas)
    image = np.zeros((len(vs), len(us)), np.float32); profondeur = np.zeros(image.shape, np.int16)
    masque = np.zeros(image.shape, bool)
    par_tranche = [np.zeros(image.shape, np.uint8) for _ in (tranches or [])]
    for b0 in range(0, len(vs), bande):
        vv, uu = np.meshgrid(vs[b0:b0 + bande] / k - r0, us / k - c0, indexing="ij")
        co = [vv, uu]
        # hors de la grille du maillage, rien n'est valide (sinon l'interpolation ramène vers 0)
        dehors = (vv < 0) | (vv > P.shape[0] - 1) | (uu < 0) | (uu > P.shape[1] - 1)
        inval = (ndi.map_coordinates(invalide.astype(np.float32), co, order=1, mode="nearest") > 0.01) | dehors
        masque[b0:b0 + bande] = inval
        if inval.all():
            continue
        p = [ndi.map_coordinates(P[..., i], co, order=1, mode="nearest") / facteur for i in range(3)]
        n = [ndi.map_coordinates(N[..., i], co, order=1, mode="nearest") for i in range(3)]
        nn = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) + 1e-9
        n = [a / nn for a in n]
        ok = ~inval
        lo = [int(np.floor(a[ok].min())) - T - 2 for a in p]; hi = [int(np.ceil(a[ok].max())) + T + 3 for a in p]
        taille_go = np.prod([h_ - l_ for l_, h_ in zip(lo, hi)]) / 1e9
        if taille_go > 6:
            raise MemoryError(f"sous-volume de {taille_go:.0f} Go pour la fenêtre {fenetre} : réduire la fenêtre")
        sub = V.lire_region(chemin_volume, niveau, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), base=base, fils=24, cache=cache)
        org = [max(0, lo[2]), max(0, lo[1]), max(0, lo[0])]
        acc = np.zeros(vv.shape, np.float32); arg = np.zeros(vv.shape, np.int16); somme = np.zeros(vv.shape, np.float32)
        tr = [np.zeros(vv.shape, np.float32) for _ in par_tranche]
        for t in range(-T, T + 1):
            s = ndi.map_coordinates(sub, [p[2] + t * n[2] - org[0], p[1] + t * n[1] - org[1], p[0] + t * n[0] - org[2]], order=1, mode="constant", output=np.float32)
            mieux = s > acc; acc[mieux] = s[mieux]; arg[mieux] = t; somme += s
            for i, (t0, t1) in enumerate(tranches or []):
                if t0 <= t <= t1:
                    np.maximum(tr[i], s, out=tr[i])
        image[b0:b0 + bande] = acc if stat == "max" else somme / (2 * T + 1)
        profondeur[b0:b0 + bande] = arg
        for i in range(len(par_tranche)):
            par_tranche[i][b0:b0 + bande] = np.clip(tr[i], 0, 255).astype(np.uint8)
        print(f"  bande {b0 // bande + 1}/{-(-len(vs) // bande)} : sous-volume {sub.shape}", flush=True)
    image[masque] = 0
    for a in par_tranche:
        a[masque] = 0
    if tranches is not None:
        return image, profondeur, par_tranche
    return image, profondeur


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True); ap.add_argument("--niveau", type=int, default=0)
    ap.add_argument("--facteur", type=float, default=1.0); ap.add_argument("--maillage", required=True)
    ap.add_argument("--fenetre", type=int, nargs=4, required=True, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--pas", type=int, default=2); ap.add_argument("--demi-epaisseur", type=int, default=24)
    ap.add_argument("--stat", choices=("max", "moyenne"), default="max"); ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    img, prof = rendre(a.volume, a.niveau, a.facteur, a.maillage, tuple(a.fenetre), a.pas, a.demi_epaisseur, stat=a.stat)
    np.save(pathlib.Path(a.sortie).with_suffix(".npy"), img.astype(np.uint8)); np.save(pathlib.Path(a.sortie).with_suffix(".prof.npy"), prof.astype(np.int8))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(a.sortie)
    print("->", a.sortie, img.shape)
