#!/usr/bin/env python3
"""Une même zone de PHerc. Paris 4 (là où l'on attend le titre) vue par toutes les techniques essayées.

Zone : x 285-345 mm, y 48-106 mm du segment 20260701183124-w010-027 (juste après la dernière colonne).
Sorties :
  apercus/paris4_zone_titre_comparaison.jpg     planche : un panneau par technique, avec dessous un morceau de vrai
                                               texte vu par la même technique, à la même échelle et aux mêmes réglages
  apercus/comparaison_titre/index.html          visionneuse : on passe d'une technique à l'autre au même endroit

  .venv/bin/python -I scripts/comparaison_titre.py
"""
import glob
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import mosaique_balayage as M  # noqa: E402
import volume as V  # noqa: E402

PX = 0.0096                                     # mm par pixel des calques
ZONE = (48.0, 106.0, 285.0, 345.0)              # y0, y1, x0, x1 en mm
REF = (24.0, 40.0, 218.0, 246.0)                # vrai texte : fin de la dernière colonne
REF_FIN = (60.0, 76.0, 174.0, 198.0)            # vrai texte couvert par le scan 1,129 µm : avant-dernière colonne
SEG = V.DATA / "PHercParis4/segments"
APERCUS = V.RACINE / "apercus"
I9 = V.DATA / "ink9"


def police(t):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, t)
        except Exception:
            pass
    return ImageFont.load_default()


def etendue(z):
    return int(round(z[0] / PX)), int(round(z[1] / PX)), int(round(z[2] / PX)), int(round(z[3] / PX))


def carte3d(nom, z):
    r0, r1, c0, c1 = etendue(z); d = C.X0 // C.PAS
    return np.asarray(np.load(C.SORTIE / nom, mmap_mode="r")[r0:r1, c0 - d:c1 - d]).astype(np.float32) / 255


def canonique(z):
    r0, r1, c0, c1 = etendue(z)
    tif = glob.glob(str(SEG / "20260701183124-w010-027/ink-detection/*new_canon*.tif"))[0]
    a = V.lire_tif_region(tif, (r0 * 4, r1 * 4), (c0 * 4, c1 * 4)).reshape(r1 - r0, 4, c1 - c0, 4).mean((1, 3))
    return np.clip((a - 32) / (230 - 32), 0, 1)


def fin2d(z):
    """Carte 2D publiée sur le scan 1,129 µm (segment jumeau), replacée dans le repère du segment 701."""
    r0, r1, c0, c1 = etendue(z); e = 170160 / 180700
    tif = glob.glob(str(SEG / "20260623141924-w010-027/ink-detection/*1.129um*.tif"))[0]
    ya, yb = int((r0 * 4 - 4416) / e), int((r1 * 4 - 4416) / e) + 8; xa, xb = int((c0 * 4 - 40) / e), int((c1 * 4 - 40) / e) + 8
    a = V.lire_tif_region(tif, (ya, yb), (xa, xb)).astype(np.float32)
    n, m = a.shape[0] // 4, a.shape[1] // 4
    b = ndi.zoom(a[:n * 4, :m * 4].reshape(n, 4, m, 4).mean((1, 3)), e, order=1)[:r1 - r0, :c1 - c0]
    out = np.zeros((r1 - r0, c1 - c0), np.float32); out[:b.shape[0], :b.shape[1]] = b
    return np.clip((out - 32) / (230 - 32), 0, 1)


def ink9(nom, y0, x0, z):
    r0, r1, c0, c1 = etendue(z); dy, dx = int(round(y0 / PX)), int(round(x0 / PX))
    a = np.load(I9 / f"{nom}_d+0.npy", mmap_mode="r")
    return np.clip((np.asarray(a[:, r0 - dy:r1 - dy, c0 - dx:c1 - dx]).astype(np.float32).mean(0) - 64) / (230 - 64), 0, 1)


def surface(nom, y0, x0, z):
    r0, r1, c0, c1 = etendue(z); dy, dx = int(round(y0 / PX)), int(round(x0 / PX))
    a = np.asarray(np.load(I9 / f"{nom}_surface.npy", mmap_mode="r")[r0 - dy:r1 - dy, c0 - dx:c1 - dx]).astype(np.float32)
    lo, hi = np.percentile(a[::7, ::7], 3), np.percentile(a[::7, ::7], 99.5)
    return 0.85 * np.clip((a - lo) / (hi - lo), 0, 1)


def balayage(z):
    r0, r1, c0, c1 = etendue(z); m, _, _ = M.mosaique()
    dy, dx = M.Y0 // 4, M.X0 // 4
    out = np.full((r1 - r0, c1 - c0), -30, np.float32)
    a = m[max(r0 - dy, 0):r1 - dy, max(c0 - dx, 0):c1 - dx]
    out[max(dy - r0, 0):max(dy - r0, 0) + a.shape[0], max(dx - c0, 0):max(dx - c0, 0) + a.shape[1]] = a
    return np.clip((out + 2) / 6, 0, 1)                 # logit -2 (papyrus nu) -> noir ; +4 (encre franche) -> blanc


def etalon():
    z = np.load(V.DATA / "PHercParis4/balayage/etalon_texte_mosaique.npz")
    return np.clip((z["s11_a100_p"].astype(np.float32) + 2) / 6, 0, 1)


def reduire(a, f, maxi=False):
    h, w = a.shape[0] // f, a.shape[1] // f
    b = a[:h * f, :w * f].reshape(h, f, w, f)
    return b.max((1, 3)) if maxi else b.mean((1, 3))


def main():
    calques = [
        ("1. Surface du papyrus (scan 2,4 µm, sans modèle)", lambda z: surface("p4_vierge", 0, 281, z) if z == ZONE else surface("p4_texte", 5, 215, z), REF),
        ("2. Modèle 3D d'avril (organisateurs, scan 2,4 µm)", lambda z: carte3d("avril_propre.npy", z), REF),
        ("3. Modèle 3D de septembre (organisateurs, scan 2,4 µm)", lambda z: carte3d("student_propre.npy", z), REF),
        ("4. Modèle 2D « canonical » (organisateurs) — lit le titre de PHerc. 139", canonique, REF),
        ("5. Modèle 2D des organisateurs sur le scan 1,129 µm", fin2d, REF_FIN),
        ("6. Notre calcul : modèle 3D sur le scan 1,129 µm (nuit du 13)", balayage, None),
        ("7. Notre calcul : modèle multi-rouleaux ink_9um — lit le titre de PHerc. 139", lambda z: ink9("p4_vierge", 0, 281, z) if z == ZONE else ink9("p4_texte", 5, 215, z), REF),
    ]
    F = 4; ech = PX * F
    h, w = int((ZONE[1] - ZONE[0]) / ech), int((ZONE[3] - ZONE[2]) / ech)
    hr, wr = int((REF[1] - REF[0]) / ech), int((REF[3] - REF[2]) / ech)
    bande, marge = 54, 14
    cols, lignes = 4, 2
    pw, ph = w + marge, bande + h + 46 + hr + marge
    planche = Image.new("RGB", (cols * pw + marge, lignes * ph + marge), (18, 18, 18)); d = ImageDraw.Draw(planche)
    f1, f2, f3 = police(30), police(24), police(20)
    dossier = APERCUS / "comparaison_titre"; dossier.mkdir(exist_ok=True); noms = []
    for i, (titre, lire, ref) in enumerate(calques):
        a = lire(ZONE)
        Image.fromarray((255 * reduire(a, 2)).astype(np.uint8)).save(dossier / f"calque{i + 1}.jpg", quality=88); noms.append(titre)
        px, py = marge + (i % cols) * pw, marge + (i // cols) * ph
        d.text((px, py + 8), titre, fill=(255, 220, 120), font=f1 if len(titre) < 62 else f2)
        planche.paste(Image.fromarray((255 * reduire(a, F)).astype(np.uint8)), (px, py + bande))
        for xm in range(int(ZONE[2]) + 5, int(ZONE[3]), 10):
            c = px + (xm - ZONE[2]) / ech; d.line([(c, py + bande), (c, py + bande + 14)], fill=(255, 200, 0), width=2); d.text((c + 4, py + bande + 2), f"{xm} mm", fill=(255, 200, 0), font=f3)
        for ym in range(int(ZONE[0]) + 2, int(ZONE[1]), 10):
            r = py + bande + (ym - ZONE[0]) / ech; d.line([(px, r), (px + 14, r)], fill=(255, 200, 0), width=2); d.text((px + 18, r - 12), f"{ym} mm", fill=(255, 200, 0), font=f3)
        yr = py + bande + h + 6
        if ref is not None:
            b = lire(ref); d.text((px, yr + 6), "Pour comparer : du vrai texte du rouleau vu par la même technique, même échelle", fill=(140, 220, 255), font=f3)
            planche.paste(Image.fromarray((255 * reduire(b, F)).astype(np.uint8)), (px, yr + 40))
        else:
            b = etalon(); d.text((px, yr + 6), "Pour comparer : une ligne de vrai texte vue par la même technique (bande de 3 mm de haut)", fill=(140, 220, 255), font=f3)
            planche.paste(Image.fromarray((255 * reduire(b, F)).astype(np.uint8)), (px, yr + 40))
        print(titre, "| part claire (> 0,5) dans la zone : %.2f %%" % (100 * (a > 0.5).mean()), flush=True)
    px, py = marge + 3 * pw, marge + ph
    texte = ["PHerc. Paris 4 — la zone où l'on attend le titre",
             "", "Même zone dans les sept panneaux : x 285-345 mm, y 48-106 mm",
             "du segment de fin de rouleau, juste après la dernière colonne.",
             "", "Clair = le modèle voit de l'encre (panneaux 2 à 7).",
             "Le panneau 1 est le scan brut, sans modèle.",
             "", "Sous chaque panneau : un morceau de vrai texte vu par la même",
             "technique. C'est à cela que ressemblerait un titre lisible.",
             "", "Échelle : une lettre du texte mesure environ 3 mm de haut."]
    for k, t in enumerate(texte):
        d.text((px + 10, py + 60 + 44 * k), t, fill=(235, 235, 235) if k else (255, 220, 120), font=f1 if k == 0 else f2)
    planche.save(APERCUS / "paris4_zone_titre_comparaison.jpg", quality=88)
    print("planche", planche.size)
    (dossier / "calques.json").write_text(json.dumps(dict(noms=noms, zone=ZONE, mm_par_px=PX * 2), ensure_ascii=False))


if __name__ == "__main__":
    main()
