#!/usr/bin/env python3
"""Extrait, pour un bloc du segment w010-027 de PHerc. Paris 4, les deux scans à 2,4 µm
(78 keV et 137 keV) et la prédiction d'encre 3D, rangés en couches parallèles à la feuille.

Chaque bloc donne trois piles [2T+1, H, W] : ct78, ct137, encre. La couche d'indice T est
la position du maillage ; t > 0 va vers l'extérieur du rouleau. Les mêmes points 3D sont
lus dans les deux scans (la matrice de passage vient des métadonnées officielles), ce qui
permet de comparer voxel à voxel le contraste de l'encre aux deux énergies.

  .venv/bin/python -I scripts/double_energie.py extraire Y0 X0 [Y0 X0 ...]   # blocs de 3 mm
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import rendu_surface as R  # noqa: E402
import volume as V  # noqa: E402

CT78 = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
CT137 = "samples/PHercParis4/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr/"
BASE137 = "https://data.aws.ash2txt.org/"
ENCRE = C.MODELES["student"][0]              # niveau 0 = 4,8 µm
SORTIE = V.DATA / "PHercParis4/double_energie"
COTE = 1250                                   # 3 mm à 2,4 µm
T = 40                                        # ±96 µm


def points(fenetre):
    """Positions 3D (voxels du scan 78 keV) et normales pour chaque pixel du bloc."""
    (X, Y, Z), k = R._maillage(C.MAILLAGE)
    y0, y1, x0, x1 = fenetre
    r0, r1, c0, c1 = int(y0 // k) - 2, int(y1 // k) + 3, int(x0 // k) - 2, int(x1 // k) + 3
    P = np.stack([g[r0:r1, c0:c1].astype(np.float64) for g in (X, Y, Z)], -1)
    N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    vv, uu = np.meshgrid(np.arange(y0, y1) / k - r0, np.arange(x0, x1) / k - c0, indexing="ij")
    p = np.stack([ndi.map_coordinates(P[..., i], [vv, uu], order=1, mode="nearest") for i in range(3)], -1)
    n = np.stack([ndi.map_coordinates(N[..., i], [vv, uu], order=1, mode="nearest") for i in range(3)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return p, n


def empiler(p, n, chemin, niveau, facteur, base=V.BUCKET, matrice=None):
    """Échantillonne un volume aux points p + t·n, t = -T..T (en voxels de 2,4 µm)."""
    ts = np.arange(-T, T + 1)
    coins = np.concatenate([(p + t * n).reshape(-1, 3) for t in (-T, T)])
    if matrice is not None:
        coins = coins @ matrice[:3, :3].T + matrice[:3, 3]
    lo = np.floor(coins.min(0) / facteur).astype(int) - 2; hi = np.ceil(coins.max(0) / facteur).astype(int) + 3
    sub = V.lire_region(chemin, niveau, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), base=base, fils=24, cache=False)
    org = np.maximum(lo, 0)
    pile = np.zeros((len(ts),) + p.shape[:2], np.uint8)
    for i, t in enumerate(ts):
        q = p + t * n
        if matrice is not None:
            q = q @ matrice[:3, :3].T + matrice[:3, 3]
        q = q / facteur
        pile[i] = np.clip(ndi.map_coordinates(sub, [q[..., 2] - org[2], q[..., 1] - org[1], q[..., 0] - org[0]], order=1, mode="constant"), 0, 255)
    return pile


def extraire(y0, x0):
    cible = SORTIE / f"bloc_{y0:05d}_{x0:06d}.npz"
    if cible.exists():
        return cible
    SORTIE.mkdir(parents=True, exist_ok=True)
    M = np.load(V.DATA / "PHercParis4/transfo_78_vers_137.npy")
    p, n = points((y0, y0 + COTE, x0, x0 + COTE))
    ct78 = empiler(p, n, CT78, 0, 1)
    ct137 = empiler(p, n, CT137, 0, 1, base=BASE137, matrice=M)
    encre = empiler(p, n, ENCRE, 0, 2)
    tmp = SORTIE / f"bloc_{y0:05d}_{x0:06d}.part.npz"
    np.savez_compressed(tmp, ct78=ct78, ct137=ct137, encre=encre)
    tmp.rename(cible)
    print(f"bloc y={y0} x={x0} : 78 keV moy {ct78.mean():.1f}, 137 keV moy {ct137.mean():.1f}, encre>128 {100 * (encre > 128).mean():.2f} %", flush=True)
    return cible


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extraire":
        v = [int(a) for a in sys.argv[2:]]
        for y0, x0 in zip(v[::2], v[1::2]):
            extraire(y0, x0)
