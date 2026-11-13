#!/usr/bin/env python3
"""Contrôle de `ink9.py titre139` : notre sortie ressemble-t-elle à la carte d'encre publiée du titre de PHerc. 139 ?

La carte publiée par les organisateurs (modèle « canonical », réduite 8 fois : 19,2 µm par pixel) est
téléchargée une fois dans data/ ; nos cartes (9,6 µm par pixel) sont moyennées par 2 x 2 puis comparées
pixel à pixel (corrélation), pour chaque jeu de poids et pour leur moyenne.

  .venv/bin/python -I scripts/ink9.py titre139
  .venv/bin/python -I scripts/accord_titre139.py
"""
import glob
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

PUBLIEE = ("PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi/ink-detection/downsampled/"
           "PHerc0139-20260422000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg")


def reduire(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


if __name__ == "__main__":
    local = V.recuperer(PUBLIEE)
    if local is None:
        raise SystemExit("carte publiée introuvable dans le bucket : " + PUBLIEE)
    Image.MAX_IMAGE_PIXELS = None
    pub = np.asarray(Image.open(local).convert("L")).astype(np.float32)
    cartes = {pathlib.Path(f).stem.replace("titre139_", ""): np.load(f).astype(np.float32)
              for f in sorted(glob.glob(str(V.DATA / "ink9/titre139_seed*.npy")))}
    if not cartes:
        raise SystemExit("lancer d'abord : ink9.py titre139")
    moyenne = np.mean(list(cartes.values()), axis=0)
    h, w = min(moyenne.shape[0] // 2, pub.shape[0]), min(moyenne.shape[1] // 2, pub.shape[1])
    dedans = reduire(moyenne)[:h, :w] > 0                                         # hors du segment, la carte vaut zéro
    print(f"carte publiée {pub.shape}, nos cartes {moyenne.shape} ; {100 * dedans.mean():.0f} % de pixels dans le segment")
    for nom, a in list(cartes.items()) + [("moyenne des jeux de poids", moyenne)]:
        r = np.corrcoef(reduire(a)[:h, :w][dedans], pub[:h, :w][dedans])[0, 1]
        print(f"   {nom} : corrélation avec la carte publiée {r:.2f}")
    if len(cartes) >= 2:
        a, b = list(cartes.values())[:2]
        print(f"   entre les deux premiers jeux de poids : {np.corrcoef(a[a > 0][::5], b[a > 0][::5])[0, 1]:.2f}")
