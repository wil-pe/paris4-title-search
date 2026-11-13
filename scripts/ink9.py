#!/usr/bin/env python3
"""Fait tourner le modèle d'encre multi-rouleaux `scrollprize/ink_9um` sur un volume de surface public.

Ce modèle (petit étage 3D puis U-Net 2D) a été entraîné sur quatre rouleaux, surtout PHerc. 139,
à partir de volumes de surface ramenés à ~9,6 µm isotropes sur 21 couches. Il n'a jamais été
publié sur le segment de la fin de PHerc. Paris 4.

Étapes, fidèles au code officiel (externe/villa-ink9/ink-detection) :
  1. lire le niveau 2 (9,6 µm) du volume de surface à 2,4 µm, garder les 84 couches centrales
     et les moyenner par 4 -> 21 couches (scripts/prepare_9um_isotropic_input.py) ;
  2. fenêtres de 128 x 128 sur les 17 couches centrales, pas de 64, normalisation robuste par
     fenêtre, fusion par fenêtre de Hann (koine_machines/inference/infer.py) ;
  3. sortie : probabilité d'encre en uint8, un fichier .npy par jeu de poids.

  .venv/bin/python -I scripts/ink9.py titre139                    # essai sur le titre de PHerc. 139
  .venv/bin/python -I scripts/ink9.py natif NOM CHEMIN [décalages]         # volume de surface natif à ~9 µm (ex. PHerc. 800), segment entier
  .venv/bin/python -I scripts/ink9.py paris4 NOM Y0 Y1 X0 X1 [décalages]   # zone du segment 701 de Paris 4, en mm ; reprise possible
"""
import glob
import json
import os
import pathlib
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

RACINE = V.RACINE
CODE = RACINE / "externe/villa-ink9"
# jeux de poids utilisés : par défaut les deux derniers (pas 75 000) ; INK9_POIDS="*" pour les 14, ou tout autre motif
POIDS = sorted(glob.glob(str(RACINE / "modeles/ink_9um" / (os.environ.get("INK9_POIDS", "*step-075000") + ".pth"))))
SV_139 = "PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr"
SV_P4 = "PHercParis4/segments/20260701183124-w010-027/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr"
SORTIE = V.DATA / "ink9"


def _imports():
    """Rend importables le code officiel et le paquet vesuvius sans exécuter son __init__ (dépendances lourdes)."""
    if "koine_machines" in sys.modules:
        return
    sys.path.insert(0, str(CODE / "ink-detection"))
    base = CODE / "vesuvius/src"
    for nom in ("vesuvius", "vesuvius.models", "vesuvius.models.build", "vesuvius.image_proc", "vesuvius.image_proc.intensity"):
        if nom not in sys.modules:
            paquet = types.ModuleType(nom); paquet.__path__ = [str(base / nom.replace(".", "/"))]; sys.modules[nom] = paquet


def charger(poids, device):
    _imports()
    import koine_machines.inference.infer as I
    payload = torch.load(poids, map_location="cpu", weights_only=True)       # mode sûr : aucun code exécuté au chargement
    b = I.build_repo_training_model_bundle(payload, poids)
    assert b.preprocessing == "tifxyz_robust" and b.roi_size == 128, (b.preprocessing, b.roi_size)
    return b.model.to(device).eval(), b.in_chans


def entree(chemin_sv, y, x, decalage=0):
    """Volume de surface -> 21 couches à 9,6 µm. y, x : bornes en pixels du niveau 2 (9,6 µm).
    decalage : nombre de couches de 2,4 µm dont on décale la fenêtre de profondeur (essais autour de la surface)."""
    a = V.lire_region(chemin_sv, 2, (0, 109), y, x, fils=32, cache=False)
    z0 = -(-(a.shape[0] - 84) // 2) + decalage                               # 84 couches centrales, comme le script officiel
    assert 0 <= z0 and z0 + 84 <= a.shape[0], "décalage hors du volume"
    b = a[z0:z0 + 84].astype(np.float32).reshape(21, 4, a.shape[1], a.shape[2]).mean(1)
    return np.rint(b).astype(np.uint8)


def predire(vol21, modele, profondeur, device, lot=32, pas=64):
    """Probabilité d'encre (float32, H x W) ; fenêtres de 128, fusion de Hann."""
    _imports()
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    T = 128
    d0 = (vol21.shape[0] // 2) - (profondeur // 2)                           # recadrage central, comme l'entraînement
    v = vol21[d0:d0 + profondeur]
    H, W = v.shape[1:]
    ys = sorted(set(list(range(0, max(H - T, 0) + 1, pas)) + [max(H - T, 0)])); xs = sorted(set(list(range(0, max(W - T, 0) + 1, pas)) + [max(W - T, 0)]))
    hann = torch.hann_window(T, periodic=False); poids = torch.clamp(torch.outer(hann, hann), min=1e-3).numpy()
    somme = np.zeros((H, W), np.float32); norme = np.zeros((H, W), np.float32)
    coins = [(yy, xx) for yy in ys for xx in xs]
    with torch.inference_mode():
        for i in range(0, len(coins), lot):
            groupe, images = [], []
            for yy, xx in coins[i:i + lot]:
                p = np.zeros((profondeur, T, T), np.float32)
                bloc = v[:, yy:yy + T, xx:xx + T]
                if not bloc.any():
                    continue                                                 # fenêtre vide : hors du segment
                p[:, :bloc.shape[1], :bloc.shape[2]] = bloc
                groupe.append((yy, xx, bloc.shape[1], bloc.shape[2])); images.append(normalize_robust(p))
            if not groupe:
                continue
            logits = modele(torch.from_numpy(np.stack(images)[:, None]).to(device))
            pr = logits.float().sigmoid()
            if pr.shape[-2:] != (T, T):
                pr = torch.nn.functional.interpolate(pr, size=(T, T), mode="bilinear", align_corners=False)
            pr = pr.cpu().numpy()[:, 0]
            for (yy, xx, h, w), q in zip(groupe, pr):
                somme[yy:yy + h, xx:xx + w] += q[:h, :w] * poids[:h, :w]; norme[yy:yy + h, xx:xx + w] += poids[:h, :w]
    return np.where(norme > 0, somme / np.maximum(norme, 1e-6), 0).astype(np.float32)


def tourner(nom, chemin_sv, y, x, decalage=0):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    SORTIE.mkdir(parents=True, exist_ok=True)
    t = time.time(); vol = entree(chemin_sv, y, x, decalage)
    print(f"{nom} : entrée {vol.shape} lue en {time.time() - t:.0f} s (part non vide {100 * (vol[10] > 0).mean():.0f} %)", flush=True)
    np.save(SORTIE / f"{nom}_entree.npy", vol[8:13].mean(0).astype(np.uint8))            # image de la surface, pour contrôle
    for p in POIDS:
        t = time.time(); modele, prof = charger(p, device)
        pr = predire(vol, modele, prof, device)
        cle = pathlib.Path(p).stem.replace("hybrid_3d2d-", "").replace("-step-", "_")
        np.save(SORTIE / f"{nom}_{cle}.npy", np.rint(255 * pr).astype(np.uint8))
        print(f"   {cle} : {time.time() - t:.0f} s ; probabilité médiane {np.median(pr[pr > 0]):.3f}, 99 % {np.percentile(pr[pr > 0], 99):.3f}, max {pr.max():.3f}", flush=True)
        del modele
        if device.type == "mps":
            torch.mps.empty_cache()


def tourner_grand(nom, chemin_sv, y, x, decalages=(0,), tuile=2048, halo=128):
    """Grande zone, tuile par tuile (reprise possible). y, x : bornes en pixels du niveau 2 (9,6 µm).
    Sorties : data/ink9/{nom}_d{décalage}.npy, uint8, forme (nombre de jeux de poids, H, W), et {nom}_surface.npy."""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    SORTIE.mkdir(parents=True, exist_ok=True)
    H, W = y[1] - y[0], x[1] - x[0]

    def ouvrir(chemin, forme):
        return np.lib.format.open_memmap(chemin, mode="r+" if chemin.exists() else "w+", dtype=np.uint8, shape=forme)

    sorties = {d: ouvrir(SORTIE / f"{nom}_d{d:+d}.npy", (len(POIDS), H, W)) for d in decalages}
    surface = ouvrir(SORTIE / f"{nom}_surface.npy", (H, W))
    suivi = SORTIE / f"{nom}_fait.json"; fait = set(map(tuple, json.loads(suivi.read_text()))) if suivi.exists() else set()
    modeles = [charger(p, device) for p in POIDS]
    # bande centrale d'abord (la plus probable pour un titre), puis on s'en éloigne
    for ty in sorted(range(0, H, tuile), key=lambda v: abs(v + tuile / 2 - H / 2)):
        for tx in range(0, W, tuile):
            if (ty, tx) in fait:
                continue
            t = time.time()
            ya, yb, xa, xb = max(ty - halo, 0), min(ty + tuile + halo, H), max(tx - halo, 0), min(tx + tuile + halo, W)
            a = V.lire_region(chemin_sv, 2, (0, 109), (y[0] + ya, y[0] + yb), (x[0] + xa, x[0] + xb), fils=32, cache=False)
            tl = time.time() - t
            h, w = min(tuile, H - ty), min(tuile, W - tx); dedans = (slice(ty - ya, ty - ya + h), slice(tx - xa, tx - xa + w))
            surface[ty:ty + h, tx:tx + w] = a[50:59].mean(0)[dedans]
            if a.any():
                for d in decalages:
                    z0 = -(-(a.shape[0] - 84) // 2) + d
                    vol = np.rint(a[z0:z0 + 84].astype(np.float32).reshape(21, 4, a.shape[1], a.shape[2]).mean(1)).astype(np.uint8)
                    for i, (modele, prof) in enumerate(modeles):
                        sorties[d][i, ty:ty + h, tx:tx + w] = np.rint(255 * predire(vol, modele, prof, device)[dedans])
            fait.add((ty, tx)); suivi.write_text(json.dumps(sorted(fait)))
            for m in list(sorties.values()) + [surface]:
                m.flush()
            if device.type == "mps":
                torch.mps.empty_cache()
            print(f"{time.strftime('%H:%M:%S')} {nom} tuile y={ty} x={tx} : lu {tl:.0f} s, total {time.time() - t:.0f} s ({len(fait)}/{-(-H // tuile) * -(-W // tuile)})", flush=True)


def tourner_natif(nom, chemin_sv, decalages=(0,)):
    """Volume de surface natif à ~9 µm (pas de regroupement de couches) : segment entier, plusieurs fenêtres de profondeur.
    Sorties : data/ink9/{nom}_d{décalage}.npy (jeux de poids, H, W) et {nom}_surface.npy."""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    SORTIE.mkdir(parents=True, exist_ok=True)
    t = time.time(); a = V.lire_region(chemin_sv, 0, (0, 10**6), (0, 10**9), (0, 10**9), fils=32, cache=False)
    print(f"{nom} : volume {a.shape} lu en {time.time() - t:.0f} s (part non vide {100 * (a[a.shape[0] // 2] > 0).mean():.0f} %)", flush=True)
    c = a.shape[0] // 2; np.save(SORTIE / f"{nom}_surface.npy", a[c - 2:c + 3].mean(0).astype(np.uint8))
    modeles = [charger(p, device) for p in POIDS]
    for d in decalages:
        z0 = (a.shape[0] - 21) // 2 + d
        assert 0 <= z0 and z0 + 21 <= a.shape[0], "décalage hors du volume"
        t = time.time(); out = np.stack([np.rint(255 * predire(a[z0:z0 + 21], m, prof, device)) for m, prof in modeles]).astype(np.uint8)
        np.save(SORTIE / f"{nom}_d{d:+d}.npy", out)
        pr = out.astype(np.float32).mean(0) / 255
        print(f"   décalage {d:+d} couches : {time.time() - t:.0f} s ; médiane {np.median(pr[pr > 0]):.3f}, 99 % {np.percentile(pr[pr > 0], 99):.3f}, part au-dessus de 0,5 : {100 * (pr > 0.5).mean():.2f} %", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    if sys.argv[1] == "titre139":
        tourner("titre139", SV_139, (0, 1740), (0, 5220))
    elif sys.argv[1] == "segment":
        # zone d'un autre segment à 2,4 µm : NOM CHEMIN_DU_VOLUME_DE_SURFACE Y0 Y1 X0 X1 (mm) [décalages]
        y0, y1, x0, x1 = (int(round(float(v) / 0.0096)) for v in sys.argv[4:8])
        tourner_grand(sys.argv[2], sys.argv[3], (y0, y1), (x0, x1), tuple(int(v) for v in sys.argv[8].split(",")) if len(sys.argv) > 8 else (0,))
    elif sys.argv[1] == "natif":
        # volume de surface natif à ~9 µm : NOM CHEMIN_DANS_LE_BUCKET [décalages en couches]
        tourner_natif(sys.argv[2], sys.argv[3], tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0,))
    else:
        # zone du segment 701 de Paris 4 : NOM Y0 Y1 X0 X1 (mm) [décalages en couches de 2,4 µm, séparés par des virgules]
        nom = sys.argv[2]
        y0, y1, x0, x1 = (int(round(float(v) / 0.0096)) for v in sys.argv[3:7])
        decs = tuple(int(v) for v in sys.argv[7].split(",")) if len(sys.argv) > 7 else (0,)
        tourner_grand(nom, SV_P4, (y0, y1), (x0, x1), decs)
