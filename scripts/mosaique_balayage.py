#!/usr/bin/env python3
"""Assemble les tuiles du balayage du scan 1,129 µm en une seule carte, et découpe des vignettes.

  .venv/bin/python -I scripts/mosaique_balayage.py                 # carte d'ensemble -> apercus/paris4_balayage_1129.png
  .venv/bin/python -I scripts/mosaique_balayage.py candidats       # planche des meilleurs endroits du détecteur

Un pixel de la mosaïque = 9,6 µm (4 pixels du segment aplati à 2,4 µm).
"""
import glob
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

B = V.DATA / "PHercParis4/balayage"
Y0, Y1, X0, X1 = 20000, 44320, 118750, 144350
APERCUS = pathlib.Path(__file__).resolve().parent.parent / "apercus"


def mosaique(cle="s11_a100_p", vide=-30.0):
    h, w = (Y1 - Y0) // 4, (X1 - X0) // 4
    m = np.full((h, w), vide, np.float32); ok = np.zeros((h, w), bool); fait = np.zeros((h, w), bool)
    for f in glob.glob(str(B / "fin_1129/*.npz")):
        if f.endswith(".part.npz"):
            continue
        y, x = (int(v) for v in pathlib.Path(f).stem.split("_")); z = np.load(f)
        r, c = (y - Y0) // 4, (x - X0) // 4; a = z["ok"]
        fait[r:r + a.shape[0], c:c + a.shape[1]] = True
        if cle in z.files:
            m[r:r + a.shape[0], c:c + a.shape[1]] = z[cle]; ok[r:r + a.shape[0], c:c + a.shape[1]] = a
    return m, ok, fait


def gris(a, lo=-7.0, hi=3.0):
    return (255 * np.clip((a - lo) / (hi - lo), 0, 1)).astype(np.uint8)


def ensemble():
    m, ok, fait = mosaique()
    f = 4
    h, w = m.shape[0] // f, m.shape[1] // f
    petit = m[:h * f, :w * f].reshape(h, f, w, f).max((1, 3))            # maximum par bloc : garde les traits fins
    img = np.stack([gris(petit)] * 3, -1)
    pas_fait = ~fait[:h * f:f, :w * f:f]
    img[pas_fait] = (40, 40, 70)
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    for x_mm in range(290, 346, 10):
        c = (x_mm / 0.0024 - X0) / 4 / f; d.line([(c, 0), (c, 12)], fill=(255, 200, 0)); d.text((c + 3, 1), f"{x_mm} mm", fill=(255, 200, 0))
    for y_mm in range(50, 106, 10):
        r = (y_mm / 0.0024 - Y0) / 4 / f; d.line([(0, r), (12, r)], fill=(255, 200, 0)); d.text((14, r - 5), f"{y_mm} mm", fill=(255, 200, 0))
    im.save(APERCUS / "paris4_balayage_1129.png")
    print("tuiles faites : %.0f %% de la zone ; image %s" % (100 * fait.mean(), im.size))


def vignette(m, x_mm, y_mm, demi_mm=2.4):
    r = int((y_mm / 0.0024 - Y0) / 4); c = int((x_mm / 0.0024 - X0) / 4); d = int(demi_mm / 0.0096)
    out = np.full((2 * d, 2 * d), -30, np.float32)
    r0, r1, c0, c1 = max(r - d, 0), min(r + d, m.shape[0]), max(c - d, 0), min(c + d, m.shape[1])
    out[r0 - (r - d):r1 - (r - d), c0 - (c - d):c1 - (c - d)] = m[r0:r1, c0:c1]
    return out


def candidats(n=12):
    js = json.loads((B / "candidats_pales.json").read_text())["candidats"][:n]
    mp, _, _ = mosaique("s11_a100_p"); me, _, _ = mosaique("s11_a100_e")
    cote = 500; cols = 4; lignes = -(-len(js) // cols)
    planche = Image.new("RGB", (cols * (cote + 6), lignes * (2 * cote // 2 + 26)), (20, 20, 20)); d = ImageDraw.Draw(planche)
    for i, (score, nom, r, c, x_mm, y_mm) in enumerate(js):
        a = Image.fromarray(gris(vignette(mp, x_mm, y_mm))).resize((cote // 2, cote // 2), Image.LANCZOS)
        b = Image.fromarray(gris(vignette(me, x_mm, y_mm))).resize((cote // 2, cote // 2), Image.LANCZOS)
        px, py = (i % cols) * (cote + 6), (i // cols) * (cote // 2 + 26)
        planche.paste(a, (px, py + 22)); planche.paste(b, (px + cote // 2 + 2, py + 22))
        d.text((px + 4, py + 4), f"#{i + 1} score {score:.1f}  x {x_mm:.1f} mm  y {y_mm:.1f} mm  (feuille | épaisseur)", fill=(255, 220, 120))
    planche.save(APERCUS / "paris4_candidats_pales.png"); print("planche", planche.size)


if __name__ == "__main__":
    candidats() if len(sys.argv) > 1 and sys.argv[1] == "candidats" else ensemble()
