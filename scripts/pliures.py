#!/usr/bin/env python3
"""Test de la pliure : un alignement vertical de marques tombe-t-il sur un pli du rouleau ?

PHerc. Paris 4 est aplati : en coupe, chaque spire est une boucle allongée, et la feuille est pliée serré
à chaque demi-tour. Le papyrus y est fissuré, et les modèles d'encre y posent des marques qui, une fois la
feuille déroulée, s'alignent verticalement comme le ferait un bord de colonne ou une haste de lettre.
Ce script mesure la courbure de la feuille directement sur le maillage du segment (tifxyz) : pour chaque
ligne du maillage, courbure de la courbe 3D lissée, en mm⁻¹ (1 mm⁻¹ = rayon de courbure de 1 mm).
Une pliure est un maximum local de courbure au-dessus de SEUIL.

  .venv/bin/python -I scripts/pliures.py carte                         # courbure de tout le segment -> data/PHercParis4/pliures.npz, et résumé
  .venv/bin/python -I scripts/pliures.py controle X_MM Y_MM            # un candidat (repère du segment, en mm) est-il sur une pliure ?
  .venv/bin/python -I scripts/pliures.py superposer CARTE.npy Y0_MM X0_MM SORTIE.png [--fenetre Y0 Y1 X0 X1]
        # pliures en bleu sur une carte à 9,6 µm/px (par ex. une sortie de ink9.py) dont le coin haut gauche est à (Y0_MM, X0_MM)
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import rendu_surface as R  # noqa: E402
import volume as V  # noqa: E402

VOXEL_MM = 0.0024            # le maillage est exprimé en voxels du scan 2,4 µm
LISSAGE_MM = 0.25            # écart-type du lissage le long de la ligne avant de dériver
SEUIL = 1.0                  # mm⁻¹ : en dessous, la feuille est seulement courbe, pas pliée
SORTIE = V.DATA / "PHercParis4/pliures.npz"


def courbure(maillage=C.MAILLAGE, bloc=256):
    """Courbure (mm⁻¹) en chaque noeud du maillage, NaN hors du segment ; renvoie aussi le pas des noeuds en mm."""
    (X, Y, Z), k = R._maillage(maillage)
    pas = k * VOXEL_MM                                           # 0,048 mm entre deux noeuds
    K = np.full(X.shape, np.nan, np.float32)
    for r0 in range(0, X.shape[0], bloc):
        P = np.stack([g[r0:r0 + bloc] for g in (X, Y, Z)], -1).astype(np.float64) * VOXEL_MM
        vide = (P[..., 2] <= 0) | (P[..., 0] <= 0)
        for i in range(P.shape[0]):                              # les trous du maillage sont comblés par interpolation avant le lissage
            ok = ~vide[i]
            if ok.sum() > 10 and not ok.all():
                u = np.arange(P.shape[1])
                for c in range(3):
                    P[i, :, c] = np.interp(u, u[ok], P[i, ok, c])
        Q = ndi.gaussian_filter1d(P, LISSAGE_MM / pas, axis=1, mode="nearest")
        d1 = np.gradient(Q, pas, axis=1); d2 = np.gradient(d1, pas, axis=1)
        kk = np.linalg.norm(np.cross(d1, d2), axis=-1) / (np.linalg.norm(d1, axis=-1) ** 3 + 1e-12)
        kk[vide] = np.nan
        K[r0:r0 + bloc] = kk
    return K, pas


def sommets(K):
    """Masque des pliures : maxima locaux de courbure le long de chaque ligne, au-dessus du seuil."""
    k = np.nan_to_num(K, nan=0.0)
    return (k > SEUIL) & (k == ndi.maximum_filter1d(k, 21, axis=1))             # maximum sur ±0,5 mm


def charger():
    if not SORTIE.exists():
        raise SystemExit("lancer d'abord : pliures.py carte")
    z = np.load(SORTIE)
    return z["courbure"].astype(np.float32), float(z["pas_mm"])


def carte():
    K, pas = courbure()
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SORTIE, courbure=K.astype(np.float16), pas_mm=pas)
    S = sommets(K)
    ecarts, par_ligne = [], []
    for i in range(0, K.shape[0], 20):                                           # une ligne par millimètre environ
        x = np.nonzero(S[i])[0] * pas
        if len(x) > 3:
            ecarts.extend(np.diff(x)); par_ligne.append(len(x))
    ecarts = np.array(ecarts)
    ok = np.isfinite(K)
    print(f"maillage {K.shape[0]} x {K.shape[1]} noeuds tous les {1000 * pas:.0f} µm, soit {K.shape[0] * pas:.0f} x {K.shape[1] * pas:.0f} mm ; enregistré dans {SORTIE}")
    print(f"courbure : médiane {np.nanmedian(K):.2f} mm⁻¹ ; part de la feuille au-dessus de {SEUIL} mm⁻¹ : {100 * (K[ok] > SEUIL).mean():.1f} %")
    print(f"pliures : {np.median(par_ligne):.0f} par ligne (médiane) ; écart entre deux pliures voisines : médiane {np.median(ecarts):.1f} mm, "
          f"de {np.percentile(ecarts, 10):.1f} à {np.percentile(ecarts, 90):.1f} mm (10e-90e centiles)")
    for a, b in ((0, 202), (202, 281), (281, 345), (345, 408)):
        e = []
        for i in range(0, K.shape[0], 20):
            x = np.nonzero(S[i])[0] * pas; x = x[(x >= a) & (x < b)]
            e.extend(np.diff(x))
        if e:
            print(f"   x {a}-{b} mm : écart médian {np.median(e):.1f} mm")


def controle(x_mm, y_mm, rayon_mm=0.75):
    K, pas = charger()
    i, j = int(round(y_mm / pas)), int(round(x_mm / pas)); r = int(round(rayon_mm / pas))
    ligne = np.nan_to_num(K[i], nan=0.0)
    S = np.nonzero(sommets(K[max(0, i - 2):i + 3]).any(0))[0] * pas
    d = np.abs(S - x_mm).min() if len(S) else np.inf
    kmax = ligne[max(0, j - r):j + r + 1].max()
    part = (np.nan_to_num(K[:, max(0, j - r):j + r + 1], nan=0.0).max(1) > SEUIL).mean()
    print(f"x {x_mm:.1f}, y {y_mm:.1f} mm : courbure maximale à ±{rayon_mm} mm = {kmax:.2f} mm⁻¹ (seuil {SEUIL}) ; pliure la plus proche à {d:.1f} mm")
    print(f"   sur toute la hauteur du segment, {100 * part:.0f} % des lignes ont une pliure à cet x (à ±{rayon_mm} mm)")
    print("   -> SUR UNE PLIURE : une marque verticale ici n'est pas un indice d'écriture" if kmax > SEUIL else "   -> pas de pliure ici")


def superposer(chemin, y0_mm, x0_mm, sortie, fenetre=None):
    from PIL import Image
    K, pas = charger()
    a = np.load(chemin, mmap_mode="r")
    if a.ndim == 3:
        a = a[0]
    px = 0.0096
    if fenetre:
        ya, yb, xa, xb = fenetre
        a = a[max(0, round((ya - y0_mm) / px)):round((yb - y0_mm) / px), max(0, round((xa - x0_mm) / px)):round((xb - x0_mm) / px)]
        y0_mm, x0_mm = max(ya, y0_mm), max(xa, x0_mm)
    f = max(1, max(a.shape) // 2400)
    img = np.asarray(a[::f, ::f]).astype(np.float32)
    bas = np.percentile(img[img > 0], 50) if (img > 0).any() else 0.0             # le fond du modèle n'est pas à zéro : on le ramène au noir
    rvb = np.stack([np.clip((img - bas) / max(255 - bas, 1) * 255 * 1.5, 0, 255)] * 3, -1).astype(np.uint8)
    ii, jj = np.nonzero(sommets(K))
    yy = np.rint((ii * pas - y0_mm) / (px * f)).astype(int); xx = np.rint((jj * pas - x0_mm) / (px * f)).astype(int)
    hx = max(1, rvb.shape[1] // 1200); hy = int(np.ceil(pas / (px * f) / 2))       # trait continu : un noeud tous les 48 µm en hauteur
    for dy in range(-hy, hy + 1):
        for dx in range(-hx, hx + 1):
            m = (yy + dy >= 0) & (yy + dy < rvb.shape[0]) & (xx + dx >= 0) & (xx + dx < rvb.shape[1])
            rvb[yy[m] + dy, xx[m] + dx] = (70, 170, 255)
    pathlib.Path(sortie).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rvb).save(sortie)
    print(f"{sortie} : {rvb.shape[1]} x {rvb.shape[0]} px, {px * f * 1000:.1f} µm par pixel, coin haut gauche à x {x0_mm:.1f}, y {y0_mm:.1f} mm ; pliures en bleu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="mode", required=True)
    sp.add_parser("carte")
    p = sp.add_parser("controle"); p.add_argument("x_mm", type=float); p.add_argument("y_mm", type=float)
    p = sp.add_parser("superposer"); p.add_argument("carte"); p.add_argument("y0_mm", type=float); p.add_argument("x0_mm", type=float); p.add_argument("sortie")
    p.add_argument("--fenetre", nargs=4, type=float, metavar=("Y0", "Y1", "X0", "X1"))
    a = ap.parse_args()
    if a.mode == "carte":
        carte()
    elif a.mode == "controle":
        controle(a.x_mm, a.y_mm)
    else:
        superposer(a.carte, a.y0_mm, a.x0_mm, a.sortie, a.fenetre)
