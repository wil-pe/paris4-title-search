#!/usr/bin/env python3
"""Fait tourner le modèle d'encre 3D `ink_paris4_3d_4.8um_gambiarra` sur un pavé de scan.

Le réseau est celui du dépôt officiel (externe/villa, branche ink-paris4-3d-gambiarra) ;
seule la boucle d'inférence est réécrite ici pour tourner sur le GPU Apple (MPS) ou le
processeur, alors que le script officiel exige CUDA. La géométrie suit la fiche du modèle :

  - le scan est à 2,4 µm (niveau 0) ; la sortie est à 4,8 µm (un voxel de sortie = 2 voxels) ;
  - un cube lit 128³ voxels, normalisés entre le 1er et le 99e centile du pavé de 256³ qui
    l'entoure, puis moyennés par 2 avant le réseau ; il produit 64³ logits ;
  - les cubes se recouvrent de moitié et sont mélangés par une fenêtre gaussienne (sigma 8).

Les cubes n'ont pas exactement les mêmes origines que dans le calcul officiel du volume
entier : les valeurs peuvent donc différer légèrement de la prédiction publiée.
"""
import contextlib
import io
import pathlib
import sys
import time
import types

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

RACINE = pathlib.Path(__file__).resolve().parent.parent
VILLA = RACINE / "externe/villa/vesuvius/src"
POIDS = RACINE / "modeles/ink_paris4_3d_4.8um_gambiarra/model.safetensors"


def appareil():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def charger_reseau(device=None):
    """Construit le réseau avec le code officiel et charge les poids publiés (chargement strict)."""
    # On n'exécute pas vesuvius/__init__.py (il importe tout le paquet) : un paquet vide suffit
    # pour atteindre vesuvius.models.build et vesuvius.ink_detection.paris4_3d_gambiarra.student.
    if "vesuvius" not in sys.modules:
        for nom in ("vesuvius", "vesuvius.ink_detection", "vesuvius.ink_detection.paris4_3d_gambiarra"):
            paquet = types.ModuleType(nom)
            paquet.__path__ = [str(VILLA / nom.replace(".", "/"))]
            sys.modules[nom] = paquet
    from vesuvius.ink_detection.paris4_3d_gambiarra import student
    with contextlib.redirect_stdout(io.StringIO()):
        reseau = student.build_network(deterministic_pool=False)
    student.load_weights(reseau, POIDS)
    return reseau.eval().to(device or appareil())


def _centiles(parent):
    """1er et 99e centiles d'un pavé uint8, identiques à np.percentile (interpolation linéaire)."""
    h = np.bincount(parent.ravel(), minlength=256); c = np.cumsum(h); n = int(c[-1])
    out = []
    for q in (0.01, 0.99):
        k = (n - 1) * q; bas = int(np.floor(k)); haut = min(bas + 1, n - 1)
        a = int(np.searchsorted(c, bas + 1)); b = int(np.searchsorted(c, haut + 1))
        out.append(a + (b - a) * (k - bas))
    return out


def origines_cubes(forme_ct, cellules=None):
    """Origines des cubes (en voxels de sortie) pour un pavé ; `cellules` est un tableau booléen
    par cellules de 64 voxels d'entrée : on ne garde que les cubes qui en touchent une marquée."""
    forme = tuple(s // 2 for s in forme_ct)
    axes = []
    for s in forme:
        a = list(range(32, s - 96 + 1, 32))
        if a and a[-1] != s - 96:
            a.append(s - 96)
        axes.append(a)
    out = [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]
    if cellules is not None:
        out = [(z, y, x) for z, y, x in out if cellules[z // 32:z // 32 + 2, y // 32:y // 32 + 2, x // 32:x // 32 + 2].any()]
    return out


def inferer(ct, reseau, lot=4, journal=True, cellules=None):
    """ct : pavé uint8 (z, y, x) à 2,4 µm, dimensions paires. Renvoie les logits à 4,8 µm sur
    tout le pavé (NaN là où aucun cube n'a pu être calculé, c'est-à-dire à moins de 32 voxels
    de sortie des bords, ou loin de la feuille si `cellules` est donné) et le temps par cube."""
    device = next(reseau.parameters()).device
    forme = tuple(s // 2 for s in ct.shape)
    origines = origines_cubes(ct.shape, cellules)
    g = np.zeros((64,) * 3, np.float32); g[32, 32, 32] = 1
    g = gaussian_filter(g, 8, mode="constant"); g /= g.max()
    somme = np.zeros(forme, np.float64); poids = np.zeros(forme, np.float64)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(origines), lot):
            groupe = origines[i:i + lot]
            x = np.empty((len(groupe), 1, 128, 128, 128), np.float32)
            for k, (z, y, xx) in enumerate(groupe):
                parent = ct[2 * z - 64:2 * z + 192, 2 * y - 64:2 * y + 192, 2 * xx - 64:2 * xx + 192]
                lo, hi = _centiles(parent)
                cube = ct[2 * z:2 * z + 128, 2 * y:2 * y + 128, 2 * xx:2 * xx + 128].astype(np.float32)
                x[k, 0] = (np.clip(cube, lo, hi) - lo) / (hi - lo) if hi - lo > 1e-6 else 0
            logits = reseau(F.avg_pool3d(torch.from_numpy(x).to(device), 2))["ink"].float().cpu().numpy()
            for k, (z, y, xx) in enumerate(groupe):
                somme[z:z + 64, y:y + 64, xx:xx + 64] += g * logits[k, 0]
                poids[z:z + 64, y:y + 64, xx:xx + 64] += g
            if journal and (i // lot) % 25 == 0:
                fait = i + len(groupe)
                print(f"  {fait}/{len(origines)} cubes, {(time.time() - t0) / fait:.2f} s par cube", flush=True)
    par_cube = (time.time() - t0) / max(len(origines), 1)
    return np.where(poids > 1e-3, somme / np.maximum(poids, 1e-9), np.nan).astype(np.float32), par_cube


if __name__ == "__main__":
    dev = appareil()
    r = charger_reseau(dev)
    print("réseau chargé sur", dev, "|", sum(p.numel() for p in r.parameters()), "paramètres")
    for d in (dev, torch.device("cpu")):
        r = r.to(d); x = torch.rand(2, 1, 64, 64, 64, device=d)
        with torch.inference_mode():
            r(x); t = time.time()
            for _ in range(3):
                y = r(x)["ink"]
            if d.type == "mps":
                torch.mps.synchronize()
        print(f"{d} : {(time.time() - t) / 6:.3f} s par cube (réseau seul), sortie {tuple(y.shape)}")
