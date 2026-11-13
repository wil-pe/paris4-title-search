#!/usr/bin/env python3
"""Étalonnage sur un titre connu : celui de PHerc. 139 (« Φιλοδήμου / περὶ θεῶν Η »).

Pour chaque tuile de la fenêtre choisie sur le segment du titre :
  1. on lit le scan 2,4 µm autour de la feuille seulement ;
  2. on fait tourner le modèle d'encre 3D (entraîné sur Paris 4) et on projette sur la feuille ;
  3. on affaiblit l'encre dans le scan brut (son excès de densité par rapport au papyrus voisin
     est multiplié par alpha) et on relance le modèle, pour plusieurs alpha.

Les voxels d'encre sont ceux que le modèle marque sur le scan intact : on ne peut donc affaiblir
que l'encre que le modèle voit déjà.

  .venv/bin/python -I scripts/etalonnage_139.py Y0 Y1 X0 X1 [alpha ...]   # pixels de 2,4 µm du segment
"""
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modele_encre as ME  # noqa: E402
import rendu_surface as R  # noqa: E402
import volume as V  # noqa: E402

SEG = "PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi/"
MAILLAGE = V.DATA / SEG / "mesh/20260422000000-on-20260102150214-2.399um.tifxyz"
CT = "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/"
SORTIE = V.DATA / "PHerc0139/etalonnage"
TUILE = 640           # côté d'une tuile en pixels du segment (1,5 mm)
T = 40                # demi-épaisseur lue autour du maillage, en voxels (96 µm)


def points(fenetre, pas=1):
    """Positions 3D et normales (voxels du scan) pour un pixel sur `pas` de la fenêtre."""
    (X, Y, Z), k = R._maillage(MAILLAGE)
    y0, y1, x0, x1 = fenetre
    r0, c0 = max(0, int(y0 // k) - 2), max(0, int(x0 // k) - 2)
    r1, c1 = int(y1 // k) + 3, int(x1 // k) + 3
    G = [g[r0:r1, c0:c1].astype(np.float64) for g in (X, Y, Z)]
    inval = (G[2] <= 0) | (G[0] <= 0)
    P = np.stack(G, -1)
    N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    vv, uu = np.meshgrid(np.arange(y0, y1, pas) / k - r0, np.arange(x0, x1, pas) / k - c0, indexing="ij")
    p = np.stack([ndi.map_coordinates(P[..., i], [vv, uu], order=1, mode="nearest") for i in range(3)], -1)
    n = np.stack([ndi.map_coordinates(N[..., i], [vv, uu], order=1, mode="nearest") for i in range(3)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    ok = ndi.map_coordinates(inval.astype(np.float32), [vv, uu], order=1, mode="nearest") < 0.01
    return p, n, ok


def projeter(logits, lo, p, n):
    """Maximum des logits le long de la normale (t = -T..T) et profondeur de ce maximum."""
    acc = np.full(p.shape[:2], -30, np.float32); arg = np.zeros(p.shape[:2], np.int8)
    v = np.nan_to_num(logits, nan=-30.0)
    for t in range(-T, T + 1, 2):
        q = (p + t * n - 0.5) / 2
        s = ndi.map_coordinates(v, [q[..., 2] - lo[2] / 2, q[..., 1] - lo[1] / 2, q[..., 0] - lo[0] / 2], order=1, mode="constant", cval=-30)
        m = s > acc; acc[m] = s[m]; arg[m] = t
    return acc, arg


def fond_papyrus(ct, encre, f=4):
    """Densité moyenne du papyrus non encré autour de chaque point, calculée à basse résolution."""
    Z, Y, X = ct.shape
    num = np.zeros((Z // f, Y // f, X // f), np.float32); den = np.zeros_like(num)
    for z0 in range(0, Z, 64):
        c = ct[z0:z0 + 64].astype(np.float32); pap = (c > 60) & ~encre[z0:z0 + 64]
        n = c.shape[0] // f
        num[z0 // f:z0 // f + n] = (c * pap).reshape(n, f, Y // f, f, X // f, f).mean((1, 3, 5))
        den[z0 // f:z0 // f + n] = pap.reshape(n, f, Y // f, f, X // f, f).mean((1, 3, 5))
    return ndi.gaussian_filter(num, 1.5) / np.maximum(ndi.gaussian_filter(den, 1.5), 1e-3)


def affaiblir(ct, encre, fond4, alpha, f=4):
    """Dans les voxels d'encre, l'excès de densité par rapport au papyrus voisin est multiplié par alpha."""
    out = ct.copy()
    for z0 in range(0, ct.shape[0], 64):
        fond = np.repeat(np.repeat(np.repeat(fond4[z0 // f:(z0 + 64) // f], f, 0), f, 1), f, 2)
        c = ct[z0:z0 + 64].astype(np.float32); e = encre[z0:z0 + 64]
        out[z0:z0 + 64] = np.where(e, np.clip(c - (1 - alpha) * np.maximum(c - fond, 0), 0, 255), c).astype(np.uint8)
    return out


def tuile(fenetre, alphas, reseau):
    y0, y1, x0, x1 = fenetre
    cible = SORTIE / f"tuile_{y0:05d}_{x0:05d}.npz"
    if cible.exists():
        return
    p4, n4, ok4 = points(fenetre, pas=4)
    if ok4.mean() < 0.2:
        return
    pts = np.concatenate([(p4 + t * n4)[ok4] for t in (-T, -T // 2, 0, T // 2, T)])
    lo = (np.floor(pts.min(0)).astype(int) - 200) // 64 * 64                     # x, y, z
    hi = -(-(np.ceil(pts.max(0)).astype(int) + 200) // 64) * 64
    forme = tuple(int(v) for v in (hi - lo)[::-1])                               # z, y, x
    cel = np.zeros(tuple(f // 64 for f in forme), bool)
    i = ((pts - lo) // 64).astype(int)
    cel[i[:, 2], i[:, 1], i[:, 0]] = True
    origines = ME.origines_cubes(forme, cel)
    blocs = set()                                                                # blocs de 128 voxels utiles (parents de 256³)
    for z, y, x in origines:
        a = [(2 * v - 64 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        b = [(2 * v + 191 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        blocs.update((i_, j_, k_) for i_ in range(a[0], b[0] + 1) for j_ in range(a[1], b[1] + 1) for k_ in range(a[2], b[2] + 1))
    t0 = time.time()
    ct = V.lire_region(CT, 0, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), fils=32, filtre=blocs)
    t_lecture = time.time() - t0
    p, n, ok = points(fenetre, pas=2)
    res = {"ok": ok}
    t0 = time.time()
    logits, _ = ME.inferer(ct, reseau, lot=8, journal=False, cellules=cel)
    res["a100"], res["prof"] = projeter(logits, lo, p, n)
    encre = ndi.binary_dilation(np.nan_to_num(logits, nan=-30.0) > 0, iterations=1)      # à 4,8 µm
    encre = np.repeat(np.repeat(np.repeat(encre, 2, 0), 2, 1), 2, 2)
    del logits
    fond4 = fond_papyrus(ct, encre)
    for a in alphas:
        lg, _ = ME.inferer(affaiblir(ct, encre, fond4, a), reseau, lot=8, journal=False, cellules=cel)
        res[f"a{int(round(a * 100)):03d}"], _ = projeter(lg, lo, p, n)
    # scan brut projeté (maximum sur ±12 voxels autour de la profondeur de l'encre), pour l'œil
    SORTIE.mkdir(parents=True, exist_ok=True)
    tmp = SORTIE / f"tuile_{y0:05d}_{x0:05d}.part.npz"
    np.savez_compressed(tmp, **res); tmp.rename(cible)
    print(f"tuile y={y0} x={x0} : pavé {forme}, {len(origines)} cubes, {len(blocs)} blocs lus en {t_lecture:.0f} s, "
          f"modèle {time.time() - t0:.0f} s, encre {100 * (res['a100'][ok] > 0).mean():.1f} %", flush=True)


if __name__ == "__main__":
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    alphas = [float(v) for v in sys.argv[5:]]
    reseau = ME.charger_reseau()
    for ya in range(y0, y1, TUILE):
        for xa in range(x0, x1, TUILE):
            tuile((ya, min(ya + TUILE, y1), xa, min(xa + TUILE, x1)), alphas, reseau)
