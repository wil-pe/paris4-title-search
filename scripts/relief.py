#!/usr/bin/env python3
"""Chercher l'encre comme un dépôt sur la feuille (relief), sans modèle d'encre.

Le long de l'épaisseur, la feuille de papyrus est une bosse de densité ; juste devant sa face
intérieure il y a un creux (l'espace avant la spire voisine). Là où il y a de l'encre, ce creux est
en partie comblé. On mesure donc, pixel par pixel, sur le scan brut :
  - comblement : densité moyenne dans le creux devant la surface (t = -58 à -29 µm) ;
  - contraste  : ce comblement rapporté à la densité du corps de la feuille (t = +19 à +48 µm) ;
  - hauteur    : position de la surface (là où la densité franchit le milieu entre creux et feuille).

  .venv/bin/python -I scripts/relief.py NOM Y0 Y1 X0 X1      # fenêtre du segment 701 de Paris 4, en mm
"""
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import balayage as Bz  # noqa: E402
import carte_fin_paris4 as C  # noqa: E402
import etalonnage_139 as E  # noqa: E402
import volume as V  # noqa: E402

E.MAILLAGE = C.MAILLAGE
MM = 0.0024
PAS = 8                      # un pixel de carte = 19,2 µm
NIV = 2                      # niveau du scan 2,4 µm : 9,6 µm par voxel
T = 14                       # ±134 µm le long de la normale
SORTIE = V.DATA / "PHercParis4/relief"


def profils(fen):
    """Densité du scan le long de la normale : tableau (2T+1, H, W), t croissant vers l'extérieur."""
    p, n, ok = E.points(fen, pas=PAS)
    f = 2 ** NIV; q = (p - (f - 1) / 2) / f; n = n
    lo = np.floor(q.reshape(-1, 3).min(0)).astype(int) - T - 3; hi = np.ceil(q.reshape(-1, 3).max(0)).astype(int) + T + 4
    c = 128
    blocs = set()
    pts = np.concatenate([(q + t * n).reshape(-1, 3)[::7] for t in (-T - 2, 0, T + 2)])
    for dz in (-2, 2):
        for dy in (-2, 2):
            for dx in (-2, 2):
                k = np.floor((pts + (dx, dy, dz)) / c).astype(int); blocs.update(map(tuple, k[:, ::-1]))
    ct = V.lire_region(Bz.CT24, NIV, (lo[2], hi[2]), (lo[1], hi[1]), (lo[0], hi[0]), fils=32, cache=False, filtre=blocs)
    out = np.empty((2 * T + 1,) + p.shape[:2], np.float32)
    for i, t in enumerate(range(-T, T + 1)):
        out[i] = ndi.map_coordinates(ct, [q[..., 2] + t * n[..., 2] - lo[2], q[..., 1] + t * n[..., 1] - lo[1], q[..., 0] + t * n[..., 0] - lo[0]], order=1, mode="nearest")
    return out, ok


def mesures(pr):
    t = np.arange(-T, T + 1)
    creux = pr[(t >= -6) & (t <= -3)].mean(0)
    corps = pr[(t >= 2) & (t <= 5)].mean(0)
    contraste = creux - corps
    # hauteur de la surface : premier passage au-dessus du niveau médian entre creux et corps, de l'intérieur vers l'extérieur
    lo = pr[(t >= -8) & (t <= -2)].min(0); hi = pr[(t >= 0) & (t <= 6)].max(0); mi = (lo + hi) / 2
    seg = pr[(t >= -8) & (t <= 6)]; ts = t[(t >= -8) & (t <= 6)]
    au_dessus = seg > mi[None]
    k = np.argmax(au_dessus[1:] & ~au_dessus[:-1], axis=0)                 # indice juste avant le passage
    a = np.take_along_axis(seg, k[None], 0)[0]; b = np.take_along_axis(seg, (k + 1)[None], 0)[0]
    haut = (ts[k] + (mi - a) / np.maximum(b - a, 1e-3)) * 9.6              # en µm
    return dict(comblement=creux, contraste=contraste, hauteur=haut.astype(np.float32))


def auc(pos, neg):
    v = np.concatenate([pos, neg]); o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(nom, y0, y1, x0, x1):
    fen = (int(y0 / MM), int(y1 / MM), int(x0 / MM), int(x1 / MM))
    t0 = time.time(); pr, ok = profils(fen)
    m = mesures(pr)
    SORTIE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SORTIE / f"{nom}.npz", ok=ok, profil=pr.astype(np.float16), **m)
    d = C.X0 // C.PAS
    enc = np.asarray(np.load(C.SORTIE / "student_propre.npy", mmap_mode="r")[fen[0] // 4:fen[1] // 4:2, fen[2] // 4 - d:fen[3] // 4 - d:2]).astype(np.float32)
    h, w = min(enc.shape[0], ok.shape[0]), min(enc.shape[1], ok.shape[1])
    E_, N_ = (enc[:h, :w] > 128) & ok[:h, :w], (enc[:h, :w] < 30) & ok[:h, :w]
    print(f"{nom} : {pr.shape[1:]} pixels de 19,2 µm, lu et mesuré en {time.time() - t0:.0f} s ; encre (modèle 3D) sur {100 * E_.mean():.1f} % des pixels")
    for cle, v in m.items():
        v = v[:h, :w]; fond = ndi.uniform_filter(v, 61); z = v - fond                 # on retire les variations lentes (1,2 mm)
        ligne = "   %-11s" % cle
        for sig in (0, 2, 4):
            zz = ndi.gaussian_filter(z, sig) if sig else z
            ligne += "  | lissage %d px : AUC %.3f" % (sig, auc(zz[E_][::3], zz[N_][::7])) if E_.sum() > 50 else ""
        print(ligne + "   (0,5 = aucun pouvoir ; 1 = parfait)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *(float(v) for v in sys.argv[2:6]))
