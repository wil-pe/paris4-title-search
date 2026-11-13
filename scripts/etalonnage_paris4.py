#!/usr/bin/env python3
"""Étalonnage sur le texte de PHerc. Paris 4 lui-même, sur les deux scans (2,4 µm et 1,129 µm).

Pour chaque tuile de 1,5 mm : on fait tourner le modèle d'encre 3D sur le scan intact, puis sur
le même scan où l'excès de densité de l'encre a été multiplié par alpha (encre de plus en plus
pâle, alpha = 0 : encre effacée). On garde les logits projetés sur la feuille, sans seuil :
  <scan>_aXXX_p : maximum dans la tranche de la feuille (t = -32..+12 voxels de 2,4 µm)
  <scan>_aXXX_e : maximum sur toute l'épaisseur lue (±40 voxels)
Le scan 1,129 µm est rééchantillonné sur la grille du scan 2,4 µm (matrice officielle).
Rien n'est mis en cache sur le disque.

  .venv/bin/python -I scripts/etalonnage_paris4.py NOM Y0 Y1 X0 X1 [alpha ...]
"""
import pathlib
import sys
import time
import traceback

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import etalonnage_139 as E  # noqa: E402
import modele_encre as ME  # noqa: E402
import volume as V  # noqa: E402

E.MAILLAGE = C.MAILLAGE                      # E.points lit ce maillage
CT24 = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
CT11 = "PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr/"
SORTIE = V.DATA / "PHercParis4/etalonnage"
TUILE, T, TRANCHE = 640, 40, (-32, 12)


def geometrie(fenetre):
    """Pavé du scan 2,4 µm autour de la tuile, cellules proches de la feuille, blocs à lire."""
    p4, n4, ok4 = E.points(fenetre, pas=4)
    if ok4.mean() < 0.2:
        return None
    pts = np.concatenate([(p4 + t * n4)[ok4] for t in (-T, -T // 2, 0, T // 2, T)])
    lo = (np.floor(pts.min(0)).astype(int) - 200) // 64 * 64                     # x, y, z
    hi = -(-(np.ceil(pts.max(0)).astype(int) + 200) // 64) * 64
    forme = tuple(int(v) for v in (hi - lo)[::-1])                               # z, y, x
    cel = np.zeros(tuple(f // 64 for f in forme), bool)
    i = ((pts - lo) // 64).astype(int)
    cel[i[:, 2], i[:, 1], i[:, 0]] = True
    blocs = set()
    for z, y, x in ME.origines_cubes(forme, cel):
        a = [(2 * v - 64 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        b = [(2 * v + 191 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        blocs.update((i_, j_, k_) for i_ in range(a[0], b[0] + 1) for j_ in range(a[1], b[1] + 1) for k_ in range(a[2], b[2] + 1))
    return lo, hi, forme, cel, blocs


def lire_11(lo, hi):
    """Scan 1,129 µm (niveau 1) rééchantillonné aux centres des voxels de 2,4 µm du pavé."""
    M = np.load(V.DATA / "PHercParis4/transfo_24_vers_1129.npy")
    coins = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ M.T
    q = (coins[:, :3] - 0.5) / 2
    a = np.floor(q.min(0)).astype(int) - 3; b = np.ceil(q.max(0)).astype(int) + 4
    sub = V.lire_region(CT11, 1, (a[2], b[2]), (a[1], b[1]), (a[0], b[0]), fils=32, cache=False)
    org = np.maximum(a, 0)
    out = np.zeros((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    yy, xx = np.meshgrid(np.arange(lo[1], hi[1], dtype=np.float64), np.arange(lo[0], hi[0], dtype=np.float64), indexing="ij")
    A, t = M[:3, :3], M[:3, 3]
    for z in range(lo[2], hi[2]):
        qx = (A[0, 0] * xx + A[0, 1] * yy + A[0, 2] * z + t[0] - 0.5) / 2 - org[0]
        qy = (A[1, 0] * xx + A[1, 1] * yy + A[1, 2] * z + t[1] - 0.5) / 2 - org[1]
        qz = (A[2, 0] * xx + A[2, 1] * yy + A[2, 2] * z + t[2] - 0.5) / 2 - org[2]
        out[z - lo[2]] = np.clip(ndi.map_coordinates(sub, [qz, qy, qx], order=1, mode="constant"), 0, 255)
    return out


def projeter(logits, lo, p, n, t0, t1):
    acc = np.full(p.shape[:2], -30, np.float32)
    v = np.nan_to_num(logits, nan=-30.0)
    for t in range(t0, t1 + 1, 2):
        q = (p + t * n - 0.5) / 2
        s = ndi.map_coordinates(v, [q[..., 2] - lo[2] / 2, q[..., 1] - lo[1] / 2, q[..., 0] - lo[0] / 2], order=1, mode="constant", cval=-30)
        np.maximum(acc, s, out=acc)
    return acc.astype(np.float16)


def serie(ct, cel, lo, p, n, alphas, reseau, nom, res):
    """Modèle sur le scan intact puis sur chaque version affaiblie ; remplit `res`."""
    logits, _ = ME.inferer(ct, reseau, lot=8, journal=False, cellules=cel)
    res[f"{nom}_a100_p"] = projeter(logits, lo, p, n, *TRANCHE); res[f"{nom}_a100_e"] = projeter(logits, lo, p, n, -T, T)
    if not alphas:
        return
    encre = ndi.binary_dilation(np.nan_to_num(logits, nan=-30.0) > 0, iterations=1)
    encre = np.repeat(np.repeat(np.repeat(encre, 2, 0), 2, 1), 2, 2)
    del logits
    fond4 = E.fond_papyrus(ct, encre)
    for a in alphas:
        lg, _ = ME.inferer(E.affaiblir(ct, encre, fond4, a), reseau, lot=8, journal=False, cellules=cel)
        cle = f"{nom}_a{int(round(a * 100)):03d}"
        res[cle + "_p"] = projeter(lg, lo, p, n, *TRANCHE); res[cle + "_e"] = projeter(lg, lo, p, n, -T, T)
        del lg


def tuile(nom, fenetre, alphas, reseau):
    y0, y1, x0, x1 = fenetre
    cible = SORTIE / f"{nom}_{y0:05d}_{x0:06d}.npz"
    if cible.exists():
        return
    g = geometrie(fenetre)
    if g is None:
        return
    lo, hi, forme, cel, blocs = g
    p, n, ok = E.points(fenetre, pas=2)
    res = {"ok": ok}
    t0 = time.time()
    ct = V.lire_region(CT24, 0, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), fils=32, filtre=blocs, cache=False)
    t1 = time.time(); serie(ct, cel, lo, p, n, alphas, reseau, "s24", res); del ct
    t2 = time.time(); ct = lire_11(lo, hi)
    nul = float((ct[::4, ::4, ::4] == 0).mean())
    t3 = time.time()
    if nul < 0.5:
        serie(ct, cel, lo, p, n, alphas, reseau, "s11", res)
    del ct
    SORTIE.mkdir(parents=True, exist_ok=True)
    tmp = SORTIE / f"{nom}_{y0:05d}_{x0:06d}.part.npz"
    np.savez_compressed(tmp, **res); tmp.rename(cible)
    print(f"{nom} y={y0} x={x0} : pavé {forme} | 2,4 µm lu en {t1 - t0:.0f} s, modèle {t2 - t1:.0f} s | 1,129 µm lu en {t3 - t2:.0f} s "
          f"(vide {100 * nul:.0f} %), modèle {time.time() - t3:.0f} s | encre 2,4 : {100 * (res['s24_a100_p'][ok] > 0).mean():.1f} %", flush=True)


if __name__ == "__main__":
    nom = sys.argv[1]
    y0, y1, x0, x1 = (int(v) for v in sys.argv[2:6])
    alphas = [float(v) for v in sys.argv[6:]]
    reseau = ME.charger_reseau()
    for ya in range(y0, y1, TUILE):
        for xa in range(x0, x1, TUILE):
            try:
                tuile(nom, (ya, min(ya + TUILE, y1), xa, min(xa + TUILE, x1)), alphas, reseau)
            except Exception:
                print(f"ÉCHEC {nom} y={ya} x={xa}"); traceback.print_exc(); sys.stdout.flush()
