#!/usr/bin/env python3
"""Planches d'examen pour la carte sans fantômes de PHerc. Paris 4.

1. Une vue d'ensemble de la zone finale : projection épaisse, encre « propre », résidu,
   avec les candidats numérotés.
2. Une planche par candidat : les mêmes couches recadrées, plus ce que voient les spires
   voisines au même endroit.

  .venv/bin/python -I scripts/planches_candidats.py [nombre de candidats, défaut 12]
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402

D = C.SORTIE
APERCUS = pathlib.Path(__file__).resolve().parent.parent / "apercus"
PAS = C.PAS
UM = 2.4 * PAS


def reduire(a, f):
    """Réduction par maximum (un trait fin reste visible)."""
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).max((1, 3))


def police(taille):
    """Police système avec accents (la police par défaut de Pillow n'en a pas)."""
    for chemin in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(chemin, taille)
        except OSError:
            pass
    return ImageFont.load_default()


def etiquette(im, texte, taille=22):
    d = ImageDraw.Draw(im)
    f = police(taille)
    d.rectangle([0, 0, 14 + d.textlength(texte, font=f), taille + 10], fill=0)
    d.text((6, 4), texte, fill=255 if im.mode == "L" else (255, 255, 255), font=f)
    return im


def vue_ensemble(cands, f=12):
    couches = [("projection épaisse (±134 µm), accord des deux modèles", None),
               ("encre propre (tranche de la feuille), accord des deux modèles", "propre_accord.npy"),
               ("après retrait des zones expliquées par un vis-à-vis", "residu.npy")]
    x0 = (84000 - C.X0) // PAS
    ims = []
    for titre, fichier in couches:
        if fichier is None:
            a = np.sqrt(np.load(D / "student_img.npy", mmap_mode="r")[:, x0:].astype(np.float32) * np.load(D / "avril_img.npy", mmap_mode="r")[:, x0:].astype(np.float32)).astype(np.uint8)
        else:
            a = np.load(D / fichier, mmap_mode="r")[:, x0:]
        im = Image.fromarray(reduire(np.asarray(a), f))
        if fichier == "residu.npy":
            im = im.convert("RGB"); d = ImageDraw.Draw(im)
            for i, c in enumerate(cands, 1):
                ya, yb, xa, xb = c["boite"]
                d.rectangle([(xa - x0) / f - 4, ya / f - 4, (xb - x0) / f + 4, yb / f + 4], outline=(255, 80, 80), width=2)
                d.text(((xb - x0) / f + 6, ya / f - 6), str(i), fill=(255, 220, 0), font=police(20))
        ims.append(etiquette(im.convert("RGB"), titre))
    w, h = ims[0].size
    P = Image.new("RGB", (w, 3 * h + 20), (70, 70, 70))
    for i, im in enumerate(ims):
        P.paste(im, (0, i * (h + 10)))
        im.save(APERCUS / f"paris4_carte_fin_{i + 1}.png")
    P.save(APERCUS / "paris4_carte_fin_ensemble.png")
    print("vue d'ensemble :", P.size, "échelle %.0f µm/px" % (UM * f), "| x de 201,6 à 408 mm, hauteur 167 mm")


def planche(i, c, marge=220):
    ya, yb, xa, xb = c["boite"]
    y0, y1 = max(0, ya - marge), yb + marge
    x0, x1 = max(0, xa - marge), xb + marge
    noms = [("encre propre, modèle sept.", "student_propre.npy"), ("encre propre, modèle avril", "avril_propre.npy"),
            ("côté voisine extérieure (t > 0), sept.", "student_dehors.npy"), ("côté voisine intérieure (t < 0), sept.", "student_dedans.npy"),
            ("projection épaisse, sept.", "student_img.npy"), ("résidu retenu", "residu.npy"),
            ("encre propre de la voisine extérieure, ramenée ici", "fantome_ext.npy"), ("encre propre de la voisine intérieure, ramenée ici", "fantome_int.npy")]
    tuiles = []
    for titre, fichier in noms:
        a = np.asarray(np.load(D / fichier, mmap_mode="r")[y0:y1, x0:x1])
        tuiles.append((titre, a))
    h, w = tuiles[0][1].shape
    ech = min(1.0, 990 / w)
    tw, th = int(w * ech), int(h * ech)
    P = Image.new("L", (2 * tw + 10, 4 * (th + 10)), 70)
    for k, (titre, a) in enumerate(tuiles):
        im = etiquette(Image.fromarray(a).resize((tw, th), Image.LANCZOS), titre, 18)
        d = ImageDraw.Draw(im)
        mm = 1000 / UM * ech
        d.rectangle([tw - 14 - mm, th - 16, tw - 14, th - 11], fill=255); d.text((tw - 14 - mm, th - 40), "1 mm", fill=255, font=police(18))
        P.paste(im, ((k % 2) * (tw + 10), (k // 2) * (th + 10)))
    nom = APERCUS / "candidats" / f"candidat_{i:02d}.png"
    nom.parent.mkdir(parents=True, exist_ok=True)
    P.save(nom)
    return nom


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cands = json.loads((D / "candidats.json").read_text())
    vue_ensemble(cands[:40])
    for i, c in enumerate(cands[:n], 1):
        print(planche(i, c).name, "x=%.1f mm y=%.1f mm" % (c["x_mm"], c["y_mm"]))
