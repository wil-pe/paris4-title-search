#!/usr/bin/env python3
"""Pour chaque point d'un segment, trouve la portion de feuille qui lui fait face de chaque côté.

Dans un rouleau comprimé, les spires voisines passent à 0,1 mm : leur encre déborde dans les
projections épaisses et crée des « fantômes ». Ce script calcule, sur une grille grossière
(un point tous les 192 µm), la position (u, v) du vis-à-vis côté extérieur (t > 0) et côté
intérieur (t < 0), ainsi que la distance 3D qui les sépare.

  .venv/bin/python -I scripts/voisinage.py <dossier tifxyz> <sortie.npz>

Sortie : pour chaque côté, u_voisin et v_voisin (pixels pleine résolution du segment),
dist (µm), le long_normale (µm) ; NaN là où aucun vis-à-vis n'est trouvé à moins de 400 µm.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rendu_surface as R  # noqa: E402

GROS = 4            # sous-échantillonnage de la grille tifxyz pour la recherche
ECART_MIN_MM = 5.0  # on ignore ce qui est à moins de 5 mm le long de la feuille (même spire)
ECART_MAX_MM = 32.0
VOXEL_UM = 2.4


def voisinage(dossier, c_debut=0):
    (X, Y, Z), k = R._maillage(dossier)
    pas_um = k * VOXEL_UM                                      # 48 µm entre deux noeuds
    P = np.stack([X, Y, Z], -1).astype(np.float32)
    P[(Z <= 0) | (X <= 0)] = np.nan
    Pg = P[::GROS, ::GROS]                                     # grille grossière
    N = np.cross(np.gradient(Pg, axis=1), np.gradient(Pg, axis=0))
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    h, w = Pg.shape[:2]
    dc_min = int(ECART_MIN_MM * 1000 / (pas_um * GROS)); dc_max = int(ECART_MAX_MM * 1000 / (pas_um * GROS))
    res = {s: dict(d2=np.full((h, w), np.inf, np.float32), dr=np.zeros((h, w), np.int16), dc=np.zeros((h, w), np.int16)) for s in (+1, -1)}
    for dc in list(range(-dc_max, -dc_min + 1)) + list(range(dc_min, dc_max + 1)):
        for dr in range(-3, 4):
            S = np.full_like(Pg, np.nan)
            ra, rb = max(0, -dr), min(h, h - dr); ca, cb = max(0, -dc), min(w, w - dc)
            S[ra:rb, ca:cb] = Pg[ra + dr:rb + dr, ca + dc:cb + dc]
            D = S - Pg
            d2 = (D ** 2).sum(-1); cote = (D * N).sum(-1)
            for s in (+1, -1):
                m = (cote * s > 0) & (d2 < res[s]["d2"])
                res[s]["d2"][m] = d2[m]; res[s]["dr"][m] = dr; res[s]["dc"][m] = dc
    # affinage sur la grille fine autour de la solution grossière
    out = {}
    rr, cc = np.meshgrid(np.arange(h) * GROS, np.arange(w) * GROS, indexing="ij")
    H, W = P.shape[:2]
    for s, nom in ((+1, "ext"), (-1, "int")):
        r1 = rr + res[s]["dr"].astype(np.int32) * GROS; c1 = cc + res[s]["dc"].astype(np.int32) * GROS
        best = np.full((h, w), np.inf, np.float32); br = r1.copy(); bc = c1.copy()
        for a in range(-GROS, GROS + 1):
            for b in range(-GROS, GROS + 1):
                r2 = np.clip(r1 + a, 0, H - 1); c2 = np.clip(c1 + b, 0, W - 1)
                d2 = ((P[r2, c2] - Pg) ** 2).sum(-1)
                m = d2 < best
                best[m] = d2[m]; br[m] = r2[m]; bc[m] = c2[m]
        dist = np.sqrt(best) * VOXEL_UM
        D = P[np.clip(br, 0, H - 1), np.clip(bc, 0, W - 1)] - Pg
        le_long = (D * N).sum(-1) * VOXEL_UM
        ok = np.isfinite(dist) & (dist < 400)
        out[f"u_{nom}"] = np.where(ok, bc * k, np.nan).astype(np.float32)
        out[f"v_{nom}"] = np.where(ok, br * k, np.nan).astype(np.float32)
        out[f"dist_{nom}"] = np.where(ok, dist, np.nan).astype(np.float32)
        out[f"normale_{nom}"] = np.where(ok, le_long, np.nan).astype(np.float32)
    out["pas_px"] = np.float32(GROS * k)                       # 80 px pleine résolution entre deux points
    return out


if __name__ == "__main__":
    o = voisinage(sys.argv[1])
    np.savez_compressed(sys.argv[2], **o)
    for nom in ("ext", "int"):
        d = o[f"dist_{nom}"]; ok = np.isfinite(d)
        print(f"côté {nom} : vis-à-vis trouvé pour {100 * ok.mean():.0f} % des points ; distance médiane {np.nanmedian(d):.0f} µm, "
              f"1er décile {np.nanpercentile(d, 10):.0f} µm")
