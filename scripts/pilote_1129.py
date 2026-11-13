#!/usr/bin/env python3
"""Pilote : le modèle d'encre 3D tourne-t-il sur le scan 1,129 µm de PHerc. Paris 4 ?

Sur un pavé centré sur une fenêtre du segment w010-027 :
  1. on lit le scan 2,4 µm (78 keV) et on y fait tourner le modèle ;
  2. on lit le scan 1,129 µm (niveau 1 = 2,26 µm), on le rééchantillonne sur la même grille de
     2,4 µm grâce à la matrice de passage officielle, et on y fait tourner le même modèle ;
  3. on compare les deux résultats entre eux et à la prédiction publiée par le challenge.

  .venv/bin/python -I scripts/pilote_1129.py NOM Y0 Y1 X0 X1     # fenêtre en pixels du segment
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import double_energie as DE  # noqa: E402
import modele_encre as ME  # noqa: E402
import volume as V  # noqa: E402

CT1129 = "PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr/"
PUBLIE = C.MODELES["student"][0]                  # niveau 0 = 4,8 µm, même grille que nos sorties
SORTIE = V.DATA / "PHercParis4/pilote"
MARGE = 160                                        # voxels de 2,4 µm autour de la fenêtre


def pave(fenetre):
    """Pavé (z0, y0, x0, z1, y1, x1) en voxels de 2,4 µm contenant la fenêtre, aligné sur 64."""
    p, _ = DE.points(fenetre)
    lo = np.floor(p.reshape(-1, 3).min(0)).astype(int) - MARGE; hi = np.ceil(p.reshape(-1, 3).max(0)).astype(int) + MARGE
    lo = lo // 64 * 64; hi = -(-hi // 64) * 64
    return (lo[2], lo[1], lo[0], hi[2], hi[1], hi[0])


def lire_1129_sur_grille(boite, M):
    """Scan 1,129 µm (niveau 1) rééchantillonné aux centres des voxels de 2,4 µm du pavé."""
    z0, y0, x0, z1, y1, x1 = boite
    coins = np.array([[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]) @ M.T
    q = (coins[:, :3] - 0.5) / 2                                   # indices au niveau 1
    lo = np.floor(q.min(0)).astype(int) - 3; hi = np.ceil(q.max(0)).astype(int) + 4
    sub = V.lire_region(CT1129, 1, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), fils=32)
    org = np.maximum(lo, 0)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float64), np.arange(x0, x1, dtype=np.float64), indexing="ij")
    A, b = M[:3, :3], M[:3, 3]
    for z in range(z0, z1):
        qx = (A[0, 0] * xx + A[0, 1] * yy + A[0, 2] * z + b[0] - 0.5) / 2 - org[0]
        qy = (A[1, 0] * xx + A[1, 1] * yy + A[1, 2] * z + b[1] - 0.5) / 2 - org[1]
        qz = (A[2, 0] * xx + A[2, 1] * yy + A[2, 2] * z + b[2] - 0.5) / 2 - org[2]
        out[z - z0] = np.clip(ndi.map_coordinates(sub, [qz, qy, qx], order=1, mode="constant"), 0, 255)
    return out, sub.shape


def projeter(vol, origine, pas_vox, fenetre, t0, t1):
    """Maximum de `vol` le long de la normale, pour t de t0 à t1 voxels de 2,4 µm, un pixel sur deux."""
    p, n = DE.points(fenetre)
    p, n = p[::2, ::2], n[::2, ::2]
    acc = np.full(p.shape[:2], -np.inf, np.float32)
    for t in range(t0, t1 + 1, 2):
        q = (p + t * n - 0.5 * (pas_vox - 1)) / pas_vox
        s = ndi.map_coordinates(vol, [q[..., 2] - origine[0], q[..., 1] - origine[1], q[..., 0] - origine[2]], order=1, mode="constant", cval=-20)
        np.maximum(acc, s, out=acc)
    return acc


def main(nom, fenetre):
    SORTIE.mkdir(parents=True, exist_ok=True)
    M = np.load(V.DATA / "PHercParis4/transfo_24_vers_1129.npy")
    boite = pave(fenetre); z0, y0, x0, z1, y1, x1 = boite
    forme = (z1 - z0, y1 - y0, x1 - x0)
    print(f"pavé {boite} : {forme} voxels de 2,4 µm = {forme[0] * 0.0024:.1f} x {forme[1] * 0.0024:.1f} x {forme[2] * 0.0024:.1f} mm", flush=True)
    t = time.time(); ct24 = V.lire_region(DE.CT78, 0, (z0, z1), (y0, y1), (x0, x1), fils=32)
    print(f"scan 2,4 µm lu en {time.time() - t:.0f} s (moyenne {ct24.mean():.1f})", flush=True)
    t = time.time(); ct11, f11 = lire_1129_sur_grille(boite, M)
    print(f"scan 1,129 µm lu et rééchantillonné en {time.time() - t:.0f} s (bloc source {f11}, moyenne {ct11.mean():.1f}, part nulle {100 * (ct11 == 0).mean():.1f} %)", flush=True)
    r = ME.charger_reseau()
    res = {}
    for cle, ct in (("24", ct24), ("1129", ct11)):
        t = time.time(); logits, par_cube = ME.inferer(ct, r, lot=8)
        res[cle] = logits; np.save(SORTIE / f"{nom}_logits_{cle}.npy", logits)
        print(f"modèle sur le scan {cle} : {time.time() - t:.0f} s au total, {par_cube:.3f} s par cube", flush=True)
    pub = V.lire_region(PUBLIE, 0, (z0 // 2, z1 // 2), (y0 // 2, y1 // 2), (x0 // 2, x1 // 2), fils=32)
    np.save(SORTIE / f"{nom}_publie.npy", pub)
    json.dump(dict(boite=[int(v) for v in boite], fenetre=list(fenetre)), open(SORTIE / f"{nom}.json", "w"))
    ok = np.isfinite(res["24"]) & np.isfinite(res["1129"])
    a, b = res["24"][ok], res["1129"][ok]; pb = pub[ok].astype(np.float32)
    print("voxels comparés :", int(ok.sum()))
    print("prédiction publiée (octets) : moyenne %.1f, part > 128 : %.2f %%" % (pb.mean(), 100 * (pb > 128).mean()))
    print("corrélation de nos logits (scan 2,4 µm) avec la prédiction publiée : %.3f" % np.corrcoef(a[::3], pb[::3])[0, 1])
    print("corrélation de nos logits (scan 1,129 µm) avec la prédiction publiée : %.3f" % np.corrcoef(b[::3], pb[::3])[0, 1])
    print("corrélation entre nos deux sorties (2,4 µm et 1,129 µm) : %.3f" % np.corrcoef(a[::3], b[::3])[0, 1])
    for cle, v in (("2,4 µm", a), ("1,129 µm", b)):
        print("scan %s : part de voxels à logit > 0 : %.2f %% ; logit médian %.1f" % (cle, 100 * (v > 0).mean(), np.median(v)))




def planche(nom):
    """Projette sur la fenêtre du segment la prédiction publiée et nos deux sorties, côte à côte."""
    from PIL import Image, ImageDraw, ImageFont
    info = json.load(open(SORTIE / f"{nom}.json")); z0, y0, x0 = info["boite"][:3]; fen = tuple(info["fenetre"])
    org = (z0 // 2, y0 // 2, x0 // 2)
    vols = [("prédiction publiée (scan 2,4 µm)", np.load(SORTIE / f"{nom}_publie.npy").astype(np.float32), None),
            ("notre calcul, scan 2,4 µm", np.load(SORTIE / f"{nom}_logits_24.npy"), "logit"),
            ("notre calcul, scan 1,129 µm", np.load(SORTIE / f"{nom}_logits_1129.npy"), "logit")]
    police = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    lignes = []
    for tranche, (t0, t1) in (("tranche de la feuille", (-32, 12)), ("côté spire voisine extérieure", (16, 56))):
        tuiles = []
        for titre, v, genre in vols:
            if genre == "logit":
                v = np.nan_to_num(v, nan=-20.0)
            img = projeter(v, org, 2, fen, t0, t1)
            img = 255 / (1 + np.exp(-img)) if genre == "logit" else img
            im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((660, 660), Image.LANCZOS); d = ImageDraw.Draw(im)
            texte = f"{titre} - {tranche}"; d.rectangle([0, 0, 12 + d.textlength(texte, font=police), 28], fill=0); d.text((6, 3), texte, fill=255, font=police)
            tuiles.append(im)
        lignes.append(tuiles)
    P = Image.new("L", (3 * 660 + 20, 2 * 660 + 10), 70)
    for i, tuiles in enumerate(lignes):
        for j, im in enumerate(tuiles):
            P.paste(im, (j * 670, i * 670))
    sortie = pathlib.Path(__file__).resolve().parent.parent / "apercus" / f"paris4_pilote_{nom}.png"
    P.save(sortie); print("->", sortie.name, P.size)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "planche":
        planche(sys.argv[2])
    else:
        main(sys.argv[1], tuple(int(v) for v in sys.argv[2:6]))
