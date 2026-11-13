#!/usr/bin/env python3
"""Balayage rapide de PHerc. Paris 4 avec le modèle d'encre 3D, sur les deux scans.

Le réseau travaille à 4,8 µm : il moyenne lui-même le scan 2,4 µm par 2. On lui donne donc
directement le niveau 1 du scan 2,4 µm (4,8 µm) ou le niveau 2 du scan 1,129 µm (4,5 µm,
rééchantillonné sur la même grille) : huit fois moins de données à télécharger. Les cubes sont
jointifs (pas de 64) au lieu de se recouvrir de moitié : huit fois moins de calcul.
`valider` mesure ce que ces deux raccourcis coûtent en fidélité.

Coordonnées « de sortie » : voxels de 4,8 µm ; l'indice j correspond au centre 2j + 0,5 du
scan 2,4 µm.

  .venv/bin/python -I scripts/balayage.py valider
  .venv/bin/python -I scripts/balayage.py etalon NOM Y0 Y1 X0 X1 0.6 0.4 0.25 0.15 0
  .venv/bin/python -I scripts/balayage.py scan NOM Y0 Y1 X0 X1 [11|24|11,24] [i/n]
"""
import gc
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import etalonnage_139 as E  # noqa: E402
import modele_encre as ME  # noqa: E402
import volume as V  # noqa: E402

E.MAILLAGE = C.MAILLAGE
CT24 = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
CT11 = "PHercParis4/volumes/20260608103018-1.129um-0.2m-78keV-masked.zarr/"
SORTIE = V.DATA / "PHercParis4/balayage"
TUILE = 1280                 # côté d'une tuile, en pixels de 2,4 µm du segment (3,1 mm)
T = 20                       # demi-épaisseur lue autour du maillage, en voxels de sortie (96 µm)
TRANCHE = (-16, 6)           # tranche de la feuille, en voxels de sortie (-77 à +29 µm)
PAS = 64                     # pas des cubes, en voxels de sortie (64 = jointifs)


def matrice_11():
    return np.load(V.DATA / "PHercParis4/transfo_24_vers_1129.npy")


def geometrie(fenetre, pas=PAS):
    """Pavé (voxels de sortie), origines des cubes proches de la feuille."""
    p4, n4, ok4 = E.points(fenetre, pas=4)
    if ok4.mean() < 0.05:
        return None
    q = np.concatenate([((p4 + 2 * t * n4)[ok4] - 0.5) / 2 for t in (-T, -T // 2, 0, T // 2, T)])
    lo = (np.floor(q.min(0)).astype(int) - 128) // 64 * 64                       # x, y, z
    hi = -(-(np.ceil(q.max(0)).astype(int) + 128) // 64) * 64
    forme = tuple(int(v) for v in (hi - lo)[::-1])                               # z, y, x
    cel = np.zeros(tuple(f // 32 for f in forme), bool)
    i = ((q - lo) // 32).astype(int)
    cel[i[:, 2], i[:, 1], i[:, 0]] = True
    axes = [list(range(32, s - 96 + 1, pas)) for s in forme]
    origines = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]
                if cel[z // 32:z // 32 + 2, y // 32:y // 32 + 2, x // 32:x // 32 + 2].any()]
    return lo, hi, forme, origines


def lire_24(lo, hi, origines):
    """Niveau 1 du scan 2,4 µm, limité aux blocs utiles."""
    blocs = set()
    for z, y, x in origines:
        a = [(v - 32 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        b = [(v + 95 + o) // 128 for v, o in zip((z, y, x), lo[::-1])]
        blocs.update((i, j, k) for i in range(a[0], b[0] + 1) for j in range(a[1], b[1] + 1) for k in range(a[2], b[2] + 1))
    return V.lire_region(CT24, 1, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), fils=32, filtre=blocs, cache=False)


def lire_11(lo, hi, origines):
    """Niveau 2 du scan 1,129 µm, rééchantillonné sur la grille de sortie, blocs utiles seulement."""
    M = matrice_11(); A, t = M[:3, :3], M[:3, 3]

    def vers_l2(xyz_sortie):                       # indices de sortie (x, y, z) -> indices du niveau 2 (x, y, z)
        return ((2 * np.asarray(xyz_sortie, float) + 0.5) @ A.T + t - 1.5) / 4

    coins = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = vers_l2(coins); a = np.floor(q.min(0)).astype(int) - 3; b = np.ceil(q.max(0)).astype(int) + 4
    blocs = set()
    for z, y, x in origines:                       # parents de 128 voxels, avec une marge
        c = np.array([[x + lo[0] + dx, y + lo[1] + dy, z + lo[2] + dz] for dx in (-34, 98) for dy in (-34, 98) for dz in (-34, 98)])
        r = vers_l2(c); u = (np.floor(r.min(0)).astype(int) - 2) // 128; w = (np.ceil(r.max(0)).astype(int) + 2) // 128
        blocs.update((k, j, i) for k in range(u[2], w[2] + 1) for j in range(u[1], w[1] + 1) for i in range(u[0], w[0] + 1))
    sub = V.lire_region(CT11, 2, (a[2], b[2]), (a[1], b[1]), (a[0], b[0]), fils=32, filtre=blocs, cache=False)
    org = np.maximum(a, 0)
    forme = (hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0])
    besoin = np.zeros(tuple(v // 32 for v in forme), bool)      # mailles de 32 voxels utiles : les parents des cubes
    for z, y, x in origines:
        besoin[(z - 32) // 32:(z + 96) // 32, (y - 32) // 32:(y + 96) // 32, (x - 32) // 32:(x + 96) // 32] = True
    out = np.zeros(forme, np.uint8)
    ys = 2 * np.arange(lo[1], hi[1], dtype=np.float64) + 0.5; xs = 2 * np.arange(lo[0], hi[0], dtype=np.float64) + 0.5

    def dalle(kz):                                  # une dalle de 32 tranches ; le reste du pavé (83 %) reste à zéro
        m2 = np.repeat(np.repeat(besoin[kz], 32, 0), 32, 1)
        if not m2.any():
            return
        ii, jj = np.nonzero(m2); yy, xx = ys[ii], xs[jj]
        bx = A[0, 0] * xx + A[0, 1] * yy + t[0] - 1.5; by = A[1, 0] * xx + A[1, 1] * yy + t[1] - 1.5; bz = A[2, 0] * xx + A[2, 1] * yy + t[2] - 1.5
        for k in range(kz * 32, kz * 32 + 32):
            zz = 2 * (k + lo[2]) + 0.5
            v = ndi.map_coordinates(sub, [(bz + A[2, 2] * zz) / 4 - org[2], (by + A[1, 2] * zz) / 4 - org[1], (bx + A[0, 2] * zz) / 4 - org[0]], order=1, mode="constant")
            out[k][ii, jj] = np.clip(v, 0, 255)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(dalle, range(besoin.shape[0])))
    return out


def inferer(ct1, reseau, origines, lot=4, compact=False):
    """ct1 : pavé à 4,8 µm. Renvoie les logits (NaN hors des cubes calculés).
    compact : logits codés sur un octet (pas de 1/16, bornés à ±7,9 ; -128 hors des cubes)."""
    device = next(reseau.parameters()).device
    compact = compact and PAS >= 64
    out = np.full(ct1.shape, -128, np.int8) if compact else np.full(ct1.shape, np.nan, np.float32)
    pds = np.zeros(ct1.shape, np.float32) if PAS < 64 else None
    if pds is not None:
        g = np.zeros((64,) * 3, np.float32); g[32, 32, 32] = 1
        g = ndi.gaussian_filter(g, 8, mode="constant"); g /= g.max(); out[:] = 0
    with torch.inference_mode():
        for i in range(0, len(origines), lot):
            groupe = origines[i:i + lot]
            x = np.zeros((lot, 1, 64, 64, 64), np.float32)       # forme fixe : sinon MPS accumule un graphe et ses tampons par taille de lot
            for k, (z, y, xx) in enumerate(groupe):
                lo, hi = ME._centiles(ct1[z - 32:z + 96, y - 32:y + 96, xx - 32:xx + 96])
                if hi - lo > 1e-6:
                    x[k, 0] = (np.clip(ct1[z:z + 64, y:y + 64, xx:xx + 64].astype(np.float32), lo, hi) - lo) / (hi - lo)
            lg = reseau(torch.from_numpy(x).to(device))["ink"].float().cpu().numpy()
            for k, (z, y, xx) in enumerate(groupe):
                if compact:
                    out[z:z + 64, y:y + 64, xx:xx + 64] = np.clip(np.rint(lg[k, 0] * 16), -127, 127)
                elif pds is None:
                    out[z:z + 64, y:y + 64, xx:xx + 64] = lg[k, 0]
                else:
                    out[z:z + 64, y:y + 64, xx:xx + 64] += g * lg[k, 0]; pds[z:z + 64, y:y + 64, xx:xx + 64] += g
    if pds is not None:
        out = np.where(pds > 1e-3, out / np.maximum(pds, 1e-9), np.nan)
    return out


def projeter(logits, lo, p, n, t0, t1):
    """Maximum le long de la normale, t en voxels de sortie ; renvoie aussi la profondeur du maximum."""
    acc = np.full(p.shape[:2], -30, np.float32); arg = np.zeros(p.shape[:2], np.int8)
    compact = logits.dtype == np.int8
    v = logits if compact else np.nan_to_num(logits, copy=False, nan=-30.0)         # sur place : le pavé peut peser plusieurs Go
    q0 = (p - 0.5) / 2 - lo
    for t in range(t0, t1 + 1):
        s = ndi.map_coordinates(v, [q0[..., 2] + t * n[..., 2], q0[..., 1] + t * n[..., 1], q0[..., 0] + t * n[..., 0]], order=1, mode="constant",
                                cval=-128 if compact else -30, output=np.float32)
        if compact:
            s /= 16
        m = s > acc; acc[m] = s[m]; arg[m] = t
    return acc.astype(np.float16), arg


def serie(ct1, origines, lo, p, n, alphas, reseau, nom, res):
    logits = inferer(ct1, reseau, origines, compact=not alphas)
    res[f"{nom}_a100_p"], _ = projeter(logits, lo, p, n, *TRANCHE)
    res[f"{nom}_a100_e"], res[f"{nom}_prof"] = projeter(logits, lo, p, n, -T, T)
    if not alphas:
        return
    encre = ndi.binary_dilation(np.nan_to_num(logits, nan=-30.0) > 0, iterations=int(os.environ.get("DILATATION", "1")))   # marge autour de l'encre à affaiblir
    del logits
    fond = E.fond_papyrus(ct1, encre, f=2)
    for a in alphas:
        lg = inferer(E.affaiblir(ct1, encre, fond, a, f=2), reseau, origines)
        cle = f"{nom}_a{int(round(a * 100)):03d}"
        res[cle + "_p"], _ = projeter(lg, lo, p, n, *TRANCHE); res[cle + "_e"], _ = projeter(lg, lo, p, n, -T, T)
        del lg


def tuile(nom, fenetre, alphas, scans, reseau):
    y0, y1, x0, x1 = fenetre
    cible = SORTIE / nom / f"{y0:05d}_{x0:06d}.npz"
    if cible.exists() or cible.with_suffix(".echec").exists():
        return False
    g = geometrie(fenetre)
    if g is None:
        return False
    lo, hi, forme, origines = g
    p, n, ok = E.points(fenetre, pas=4)                      # un pixel de carte = 9,6 µm
    res = {"ok": ok}; t0 = time.time(); msg = []
    for s in scans:
        ta = time.time()
        ct1 = lire_24(lo, hi, origines) if s == "24" else lire_11(lo, hi, origines)
        tb = time.time()
        if s == "11" and float(np.mean([(ct1[z:z + 64:4, y:y + 64:4, x:x + 64:4] == 0).mean() for z, y, x in origines])) > 0.9:
            msg.append("1,129 µm : hors couverture"); continue
        serie(ct1, origines, lo, p, n, alphas, reseau, "s" + s, res); del ct1
        msg.append(f"scan {s} : lu {tb - ta:.0f} s, modèle {time.time() - tb:.0f} s")
    cible.parent.mkdir(parents=True, exist_ok=True)
    tmp = cible.with_suffix(".part.npz")
    np.savez_compressed(tmp, **res); tmp.rename(cible)
    gpu = torch.mps.driver_allocated_memory() / 2**30 if torch.backends.mps.is_available() else 0
    print(f"{time.strftime('%H:%M:%S')} {nom} y={y0} x={x0} : pavé {forme}, {len(origines)} cubes | " + " | ".join(msg)
          + f" | total {time.time() - t0:.0f} s | GPU {gpu:.1f} Go", flush=True)
    return True


def valider():
    """Compare le chemin rapide aux logits de référence du pilote (niveau 0, cubes recouvrants)."""
    D = V.DATA / "PHercParis4/pilote"
    z0, y0, x0, z1, y1, x1 = json.load(open(D / "texte.json"))["boite"]
    lo = np.array([x0, y0, z0]) // 2; hi = np.array([x1, y1, z1]) // 2
    forme = tuple(int(v) for v in (hi - lo)[::-1])
    origines = [(z, y, x) for z in range(32, forme[0] - 95, 64) for y in range(32, forme[1] - 95, 64) for x in range(32, forme[2] - 95, 64)]
    reseau = ME.charger_reseau()
    for s, ref in (("24", "texte_logits_24.npy"), ("11", "texte_logits_1129.npy")):
        t = time.time(); ct1 = lire_24(lo, hi, origines) if s == "24" else lire_11(lo, hi, origines); tl = time.time() - t
        t = time.time(); lg = inferer(ct1, reseau, origines); tm = time.time() - t
        r = np.load(D / ref, mmap_mode="r")
        ok = np.isfinite(lg) & np.isfinite(r)
        a, b = lg[ok], np.asarray(r[ok]); A, B = a > 0, b > 0
        print(f"scan {s} : lecture {tl:.0f} s, modèle {tm:.0f} s pour {len(origines)} cubes | corrélation avec la référence {np.corrcoef(a[::5], b[::5])[0, 1]:.3f} | "
              f"accord de l'encre (Dice) {2 * (A & B).sum() / (A.sum() + B.sum()):.3f} | part d'encre : rapide {100 * A.mean():.2f} %, référence {100 * B.mean():.2f} %", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "valider":
        valider()
    else:
        nom = sys.argv[2]
        y0, y1, x0, x1 = (int(v) for v in sys.argv[3:7])
        if mode == "etalon":
            alphas, scans = [float(v) for v in sys.argv[7:]], os.environ.get("SCANS", "24,11").split(",")
        else:
            alphas, scans = [], (sys.argv[7].split(",") if len(sys.argv) > 7 else ["11"])
        part, parts = (int(v) for v in sys.argv[8].split("/")) if mode == "scan" and len(sys.argv) > 8 else (0, 1)
        maxi = int(sys.argv[9]) if mode == "scan" and len(sys.argv) > 9 else 10**9
        faites = 0
        reseau = ME.charger_reseau()
        # on commence par la bande centrale, la plus probable, puis on s'en éloigne
        rang = -1
        for ya in sorted(range(y0, y1, TUILE), key=lambda v: abs(v + TUILE / 2 - (y0 + y1) / 2)):
            for xa in range(x0, x1, TUILE):
                rang += 1
                if rang % parts != part:          # plusieurs processus se partagent les tuiles
                    continue
                try:
                    faites += bool(tuile(nom, (ya, min(ya + TUILE, y1), xa, min(xa + TUILE, x1)), alphas, scans, reseau))
                except Exception:
                    print(f"ÉCHEC {nom} y={ya} x={xa}"); traceback.print_exc(); sys.stdout.flush()
                    (SORTIE / nom).mkdir(parents=True, exist_ok=True); (SORTIE / nom / f"{ya:05d}_{xa:06d}.echec").write_text(traceback.format_exc())
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if faites >= maxi:
                    sys.exit(75)                  # il reste du travail : le lanceur relance un processus neuf
