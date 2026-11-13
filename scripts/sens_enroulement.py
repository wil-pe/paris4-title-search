#!/usr/bin/env python3
"""Sens d'enroulement de PHerc. Paris 4, mesuré sur le maillage du segment de fin (w010-027).

Deux questions, deux mesures :
  1. La fin du texte est-elle au centre du rouleau ou à l'extérieur ? On suit une ligne du maillage (hauteur
     constante) de gauche à droite, c'est-à-dire dans le sens de la lecture, et l'on mesure la distance de
     chaque point à l'axe du rouleau. Si elle diminue, le texte finit au cœur.
  2. De quel côté pointe la normale du maillage (le « t > 0 » de rendu_surface.py) ? On compare la normale
     à la direction qui s'éloigne de l'axe. Avec le profil de densité de relief.py (le dépôt d'encre est du
     côté t < 0), cela dit quelle face porte l'écriture.

L'axe est pris parallèle à l'axe z du scan ; son pied est le barycentre des 30 derniers millimètres de la
ligne (les spires les plus internes). Le rouleau étant aplati, le rayon oscille à chaque demi-tour : on
regarde donc son enveloppe par tranches de 50 mm.

  .venv/bin/python -I scripts/sens_enroulement.py [Y_MM ...]        # par défaut trois hauteurs : 42, 84 et 126 mm
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402
import rendu_surface as R  # noqa: E402

VOXEL_MM = 0.0024
DERNIERE_COLONNE = (217, 280)         # mm : étendue de la dernière colonne de texte dans le repère du segment


def mesurer(y_mm):
    (X, Y, Z), k = R._maillage(C.MAILLAGE)
    pas = k * VOXEL_MM
    i = int(round(y_mm / pas))
    P = np.stack([g[i - 1:i + 2].astype(np.float64) for g in (X, Y, Z)], -1) * VOXEL_MM       # trois lignes voisines, en mm
    ok = (P[1, :, 2] > 0) & (P[1, :, 0] > 0)
    u = np.arange(P.shape[1]) * pas
    fin = ok & (u > u[ok].max() - 30)
    centre = P[1, fin, :2].mean(0)
    rayon = np.hypot(*(P[1, :, :2] - centre).T)
    print(f"hauteur y = {y_mm:.0f} mm : ligne {i} du maillage, {ok.sum()} noeuds valides de x = {u[ok].min():.0f} à {u[ok].max():.0f} mm ; "
          f"z du scan varie de {np.ptp(P[1, ok, 2]):.2f} mm le long de la ligne")
    print("   tranche de x (mm)   rayon moyen   rayon max (mm)")
    moyennes = []
    for a in range(0, int(u[ok].max()), 50):
        m = ok & (u >= a) & (u < a + 50)
        if m.sum() > 100:
            moyennes.append(rayon[m].mean())
            print(f"   {a:4d}-{min(a + 50, int(u[ok].max())):4d}          {rayon[m].mean():5.2f}         {rayon[m].max():5.2f}")
    col = ok & (u >= DERNIERE_COLONNE[0]) & (u < DERNIERE_COLONNE[1]); apres = ok & (u >= DERNIERE_COLONNE[1])
    print(f"   dernière colonne de texte : rayon moyen {rayon[col].mean():.2f} mm ; papyrus vierge qui suit : {rayon[apres].mean():.2f} mm")
    pente = np.polyfit(u[ok], rayon[ok], 1)[0]
    print(f"   -> le rayon {'DIMINUE' if pente < 0 else 'AUGMENTE'} dans le sens de la lecture ({1000 * pente:+.1f} µm par mm) : "
          f"la fin du texte est {'au CENTRE' if pente < 0 else 'à l EXTÉRIEUR'} du rouleau")
    du = np.gradient(P[1], axis=0); dv = P[2] - P[0]                                           # même convention que rendu_surface.py
    N = np.cross(du, dv); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    radial = np.concatenate([P[1, :, :2] - centre, np.zeros((P.shape[1], 1))], 1); radial /= np.linalg.norm(radial, axis=-1, keepdims=True) + 1e-12
    loin = ok & (rayon > 1.0)                                                                  # près de l'axe, « vers l'extérieur » n'a plus de sens
    c = (N * radial).sum(-1)[loin]
    print(f"   normale du maillage : tournée vers l'extérieur sur {100 * (c > 0).mean():.0f} % des noeuds (produit scalaire médian {np.median(c):+.2f}) "
          f"-> t > 0 va vers {'l EXTÉRIEUR' if np.median(c) > 0 else 'le CENTRE'} du rouleau")


if __name__ == "__main__":
    for y in ([float(v) for v in sys.argv[1:]] or [42.0, 84.0, 126.0]):
        mesurer(y)
