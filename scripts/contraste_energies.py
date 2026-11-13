#!/usr/bin/env python3
"""Compare le contraste de l'encre à 78 keV et à 137 keV sur un bloc extrait par double_energie.py.

Pour chaque pixel on repère la profondeur de la surface encrée (maximum de la prédiction
d'encre, lissé entre voisins), puis on moyenne les deux scans autour de cette surface,
séparément pour les pixels encrés et pour les pixels nus voisins. Le contraste de l'encre
est rapporté au contraste papyrus/air, ce qui le rend indépendant de l'échelle de gris de
chaque scan.

  .venv/bin/python -I scripts/contraste_energies.py data/PHercParis4/double_energie/bloc_....npz
"""
import sys

import numpy as np
from scipy import ndimage as ndi


def profils(fichier, verbeux=True):
    z = np.load(fichier)
    a, b, e = (z[k].astype(np.float32) for k in ("ct78", "ct137", "encre"))
    T = a.shape[0] // 2
    valide = ndi.binary_erosion((b > 0).mean(0) > 0.98, iterations=20)     # zone réellement couverte par le scan 137 keV
    if valide.mean() < 0.02:
        return None
    # recalage fin dans le plan de la feuille (corrélation de phase sur la partie valide)
    ys, xs = np.nonzero(valide); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    u, v = a[T - 20:T + 20, y0:y1, x0:x1].mean(0), b[T - 20:T + 20, y0:y1, x0:x1].mean(0)
    F = np.fft.rfft2(u - u.mean()) * np.conj(np.fft.rfft2(v - v.mean())); F /= np.abs(F) + 1e-6
    c = np.fft.irfft2(F, s=u.shape); dy, dx = np.unravel_index(c.argmax(), c.shape)
    dy = dy if dy < u.shape[0] // 2 else dy - u.shape[0]; dx = dx if dx < u.shape[1] // 2 else dx - u.shape[1]
    b = np.roll(b, (dy, dx), axis=(1, 2))
    # profondeur de la surface encrée
    fort = e.max(0) > 160
    prof = e.argmax(0).astype(np.float32)
    poids = ndi.gaussian_filter((fort & valide).astype(np.float32), 25)
    surf = ndi.gaussian_filter(np.where(fort & valide, prof, 0), 25) / np.maximum(poids, 1e-4)
    fiable = valide & (poids > 0.05)
    encre = fiable & fort & (np.abs(prof - surf) < 6)
    nu = fiable & (e.max(0) < 40) & ~ndi.binary_dilation(e.max(0) > 100, iterations=6)
    # profils alignés sur la surface, de -20 à +20 voxels (2,4 µm)
    taus = np.arange(-20, 21)
    def profil(vol, masque):
        yy, xx = np.nonzero(masque); s = np.rint(surf[yy, xx]).astype(int); out = []
        for t in taus:
            k = np.clip(s + t, 0, vol.shape[0] - 1); out.append(vol[k, yy, xx].mean())
        return np.array(out)
    P = {n: profil(v, m) for n, v, m in (("78_encre", a, encre), ("78_nu", a, nu), ("137_encre", b, encre), ("137_nu", b, nu))}
    res = {"fichier": fichier, "valide": float(valide.mean()), "n_encre": int(encre.sum()), "n_nu": int(nu.sum()), "decalage": (int(dy), int(dx)), "taus": taus, **P}
    for E in ("78", "137"):
        d = P[f"{E}_encre"] - P[f"{E}_nu"]
        i = int(np.abs(d[12:29]).argmax()) + 12                      # extremum près de la surface
        papyrus = P[f"{E}_nu"].max() - P[f"{E}_nu"].min()            # amplitude papyrus / air le long du profil nu
        res[f"contraste_{E}"] = float(d[i] / papyrus); res[f"tau_{E}"] = int(taus[i]); res[f"papyrus_{E}"] = float(papyrus); res[f"delta_{E}"] = float(d[i])
    if verbeux:
        print(f"{fichier.split('/')[-1]} : couverture 137 keV {100 * res['valide']:.0f} %, {res['n_encre']} px encrés, {res['n_nu']} px nus, recalage {res['decalage']}")
        print("   tau (voxels) :", " ".join(f"{t:+4d}" for t in taus[::2]))
        for n in ("78_encre", "78_nu", "137_encre", "137_nu"):
            print(f"   {n:10s} :", " ".join(f"{x:4.0f}" for x in P[n][::2]))
        for E in ("78", "137"):
            print(f"   {E} keV : encre - nu = {res['delta_' + E]:+.1f} niveaux à tau = {res['tau_' + E]:+d}, amplitude papyrus/air {res['papyrus_' + E]:.1f} -> contraste relatif {res['contraste_' + E]:+.3f}")
        print(f"   rapport des contrastes 137 / 78 keV : {res['contraste_137'] / res['contraste_78']:.2f}")
    return res


if __name__ == "__main__":
    for f in sys.argv[1:]:
        profils(f)
