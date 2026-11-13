#!/usr/bin/env python3
"""Recherche d'un mot connu dans la carte d'encre sans fantômes de PHerc. Paris 4.

Le gabarit est découpé dans le texte du rouleau lui-même (même main, même modèle d'encre),
puis glissé sur la carte à plusieurs tailles par corrélation normalisée. Une fenêtre presque
vide ne peut pas obtenir un bon score : la variance locale a un plancher.

Les cartes utilisées sont à 38,4 µm par pixel (accord_r4.npy, residu_r4.npy).
"""
import pathlib
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carte_fin_paris4 as C  # noqa: E402

D = C.SORTIE
R4 = 16                               # pixels pleine résolution par pixel de la carte réduite
MM = 0.0024 * R4                      # 0,0384 mm par pixel
PLANCHER = 255.0 ** 2 * 0.03 * 0.97   # variance d'une fenêtre encrée à 3 %


def ncc(img, gabarit):
    """Corrélation normalisée de `gabarit` avec chaque fenêtre de `img` (centrée)."""
    t = gabarit.astype(np.float32) - gabarit.mean()
    n = t.size
    num = fftconvolve(img, t[::-1, ::-1], mode="same")
    s1 = ndi.uniform_filter(img, t.shape, mode="constant") * n
    s2 = ndi.uniform_filter(img * img, t.shape, mode="constant") * n
    var = np.maximum(s2 - s1 * s1 / n, 0) + PLANCHER * n
    return (num / (np.sqrt(var) * np.sqrt((t * t).sum()) + 1e-6)).astype(np.float32)


def chercher(img, gabarit, echelles=(1.0,)):
    """Meilleur score sur toutes les tailles ; renvoie (score, indice de la taille)."""
    meilleur = np.full(img.shape, -1, np.float32); quelle = np.zeros(img.shape, np.int8)
    for i, e in enumerate(echelles):
        g = gabarit if e == 1.0 else ndi.zoom(gabarit.astype(np.float32), e, order=1)
        s = ncc(img, g)
        m = s > meilleur
        meilleur[m] = s[m]; quelle[m] = i
    return meilleur, quelle


def pics(score, n=20, ecart=60, masque=None):
    """Les n maxima locaux les plus forts, séparés d'au moins `ecart` pixels."""
    s = score.copy()
    if masque is not None:
        s[~masque] = -1
    sortie = []
    for _ in range(n):
        i = int(s.argmax()); y, x = divmod(i, s.shape[1])
        if s[y, x] <= 0:
            break
        sortie.append((float(s[y, x]), y, x))
        s[max(0, y - ecart):y + ecart, max(0, x - ecart):x + ecart] = -1
    return sortie


def zone_sans_texte(forme):
    z = np.zeros(forme, bool)
    z[:, (84000 - C.X0) // R4:] = True
    z[:(17500 - C.Y0) // R4, (88500 - C.X0) // R4:(118500 - C.X0) // R4] = False
    return z


def mm(y, x):
    return (C.X0 + x * R4) * 0.0024, (C.Y0 + y * R4) * 0.0024     # (x mm, y mm)
