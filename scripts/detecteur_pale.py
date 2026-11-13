#!/usr/bin/env python3
"""Détecteur d'écriture pâle pour le balayage du scan 1,129 µm de PHerc. Paris 4.

Idée : on sait à quoi ressemble, pour le modèle d'encre, un vrai texte dont l'encre a été
affaiblie (étalonnage : data/PHercParis4/balayage/etalon_texte_mosaique.npz). On fabrique des
exemples « papyrus vierge + traits pâles » en superposant ces traits à de vraies tuiles vierges
du balayage, et on entraîne un petit réseau à les distinguer des mêmes tuiles sans rien. Les
lettres du dernier tiers de la ligne d'étalonnage ne servent qu'au test, pour mesurer honnêtement
jusqu'à quelle pâleur le détecteur fonctionne. On l'applique ensuite à toutes les tuiles.

  .venv/bin/python -I scripts/detecteur_pale.py            # entraîne, mesure, balaie, liste les candidats
"""
import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

B = V.DATA / "PHercParis4/balayage"
P = 160                       # côté d'une fenêtre : 1,5 mm à 9,6 µm par pixel
ALPHAS = ("a060", "a040", "a025", "a015", "a000")
rng = np.random.default_rng(0)


def norm(a):
    return (np.clip(a, -8, 6) / 4).astype(np.float32)


def charger_fonds():
    """Tuiles vierges du balayage : liste de (nom, logits, validité)."""
    out = []
    for f in sorted(glob.glob(str(B / "fin_1129/*.npz"))):
        z = np.load(f)
        if "s11_a100_p" in z.files and z["ok"].mean() > 0.9:
            out.append((pathlib.Path(f).stem, z["s11_a100_p"].astype(np.float32), z["ok"]))
    return out


def traits(mos, alpha):
    """Réponse du modèle au texte affaibli, limitée au voisinage de l'encre ; -30 ailleurs."""
    encre = ndi.binary_dilation(mos["s11_a100_p"] > 0, iterations=3)
    return np.where(encre, mos[f"s11_{alpha}_p"], -30).astype(np.float32), mos["s11_a100_p"] > 0


def exemple_positif(fond, tr, encre, x0, x1, echelle):
    """Fenêtre de fond + traits pâles pris dans la bande [x0, x1) de la ligne d'étalonnage."""
    for _ in range(50):
        h = int(P / echelle)
        r = rng.integers(0, tr.shape[0] - h + 1); c = rng.integers(x0, x1 - h + 1)
        if encre[r:r + h, c:c + h].mean() > 0.06:
            break
    t = tr[r:r + h, c:c + h]
    if echelle != 1.0:
        t = ndi.zoom(t, P / h, order=1)[:P, :P]
        t = np.pad(t, ((0, P - t.shape[0]), (0, P - t.shape[1])), constant_values=-30)
    if rng.random() < 0.5:
        t = t[:, ::-1]
    return np.maximum(fond, t)


def exemple_temoin(fond, fonds, encre, x0, x1):
    """Même montage qu'un exemple positif, mais les « traits » sont du papyrus vierge d'une autre tuile :
    mesure ce que le détecteur doit au montage lui-même plutôt qu'à de l'encre."""
    for _ in range(50):
        r = rng.integers(0, encre.shape[0] - P + 1); c = rng.integers(x0, x1 - P + 1)
        if encre[r:r + P, c:c + P].mean() > 0.06:
            break
    m = ndi.binary_dilation(encre[r:r + P, c:c + P], iterations=3)
    return np.maximum(fond, np.where(m, fenetre_fond(fonds), -30))


def fenetre_fond(fonds):
    _, a, _ = fonds[rng.integers(len(fonds))]
    r = rng.integers(0, a.shape[0] - P + 1); c = rng.integers(0, a.shape[1] - P + 1)
    w = a[r:r + P, c:c + P]
    return w[:, ::-1] if rng.random() < 0.5 else w


def reseau():
    def bloc(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(4, o), nn.ReLU())
    return nn.Sequential(bloc(1, 16), bloc(16, 32), bloc(32, 64), bloc(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))


def lot(fonds, mos_tr, x0, x1, n, alphas=ALPHAS):
    x = np.empty((n, 1, P, P), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        f = fenetre_fond(fonds)
        if i % 2:
            a = alphas[rng.integers(len(alphas))]
            tr, encre = mos_tr[a]
            f = exemple_positif(f, tr, encre, x0, x1, rng.choice([1.0, 1.0, 1.3, 1.6])); y[i] = 1
        x[i, 0] = norm(f)
    return torch.from_numpy(x), torch.from_numpy(y)


def auc(pos, neg):
    v = np.concatenate([pos, neg]); o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pli_de(nom):
    """Damier sur les tuiles : chaque tuile est notée par le réseau qui ne l'a pas vue comme exemple vierge."""
    y0, x0 = (int(v) for v in nom.split("_"))
    return ((y0 - 20000) // 1280 + (x0 - 118750) // 1280) % 2


def main():
    dev = torch.device(os.environ.get("APPAREIL", "cpu")); torch.set_num_threads(4)
    mos = dict(np.load(B / "etalon_texte_mosaique.npz"))
    mos_tr = {a: traits(mos, a) for a in ALPHAS}
    largeur = mos["ok"].shape[1]; coupe = int(largeur * 2 / 3)          # 2/3 des lettres pour apprendre, 1/3 pour tester
    fonds = charger_fonds()
    print(f"{len(fonds)} tuiles vierges utilisables")
    cands, bilan, seuils = [], {}, []
    for pli in (0, 1):
        f_app = [t for t in fonds if pli_de(t[0]) == pli]; f_test = [t for t in fonds if pli_de(t[0]) != pli]
        net = reseau().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3); perte = nn.BCEWithLogitsLoss()
        net.train()
        for pas in range(1200):
            x, y = lot(f_app, mos_tr, 0, coupe, 64)
            for g in opt.param_groups:
                g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * pas / 1200))
            opt.zero_grad(); l = perte(net(x.to(dev)).squeeze(1), y.to(dev)); l.backward(); opt.step()
        net.eval()
        with torch.inference_mode():
            # toutes les fenêtres des tuiles jamais vues, tous les 40 pixels
            fen = []
            for nom, a, ok in f_test:
                pos = [(r, c) for r in range(0, a.shape[0] - P + 1, 40) for c in range(0, a.shape[1] - P + 1, 40)]
                x = torch.from_numpy(np.stack([norm(a[r:r + P, c:c + P])[None] for r, c in pos]))
                sc = net(x.to(dev)).squeeze(1).cpu().numpy()
                y0, x0 = (int(v) for v in nom.split("_"))
                fen += [(float(v), nom, r, c, (x0 + (c + P / 2) * 4) * 0.0024, (y0 + (r + P / 2) * 4) * 0.0024) for (r, c), v in zip(pos, sc)]
            neg = np.array([v[0] for v in fen])
            seuil = float(np.quantile(neg, 0.999)); seuils.append(seuil)
            print(f"réseau {pli} : {len(f_app)} tuiles vues, {len(f_test)} jamais vues ({len(neg)} fenêtres) ; vierge : médiane {np.median(neg):.1f}, 99 % {np.quantile(neg, 0.99):.1f}, 99,9 % {seuil:.1f}, max {neg.max():.1f}")
            for a in ALPHAS:
                pos = np.concatenate([net(lot(f_test, mos_tr, coupe, largeur, 256, (a,))[0][1::2].to(dev)).squeeze(1).cpu().numpy() for _ in range(4)])
                bilan.setdefault(a, []).append((float(auc(pos, neg)), float((pos > seuil).mean()), float((pos > neg.max()).mean())))
                print(f"   encre à {int(a[1:]):3d} % : AUC {bilan[a][-1][0]:.3f} ; détectée au seuil 1/1000 : {100 * bilan[a][-1][1]:.0f} % ; au-dessus de TOUT le vierge : {100 * bilan[a][-1][2]:.0f} %")
            # test direct, sans montage : fenêtres brutes des tuiles d'étalonnage (dernier tiers de la ligne, jamais vu),
            # c'est-à-dire la vraie réponse du modèle d'encre au scan dont l'encre a été affaiblie
            for a in ("a100",) + ALPHAS:
                brut = mos[f"s11_{a}_p"].astype(np.float32)
                pos = [(r, c) for r in range(0, brut.shape[0] - P + 1, 40) for c in range(coupe, largeur - P + 1, 40)]
                sc = net(torch.from_numpy(np.stack([norm(brut[r:r + P, c:c + P])[None] for r, c in pos])).to(dev)).squeeze(1).cpu().numpy()
                bilan.setdefault("direct_" + a, []).append((float((sc > seuil).mean()), float((sc > neg.max()).mean()), float(np.median(sc))))
                print(f"   test direct, encre à {int(a[1:]):3d} % ({len(pos)} fenêtres) : au-dessus du seuil 1/1000 : {100 * (sc > seuil).mean():.0f} % ; au-dessus de TOUT le vierge : {100 * (sc > neg.max()).mean():.0f} % ; score médian {np.median(sc):.1f}")
            xt = np.stack([norm(exemple_temoin(fenetre_fond(f_test), f_test, mos["s11_a100_p"] > 0, coupe, largeur))[None] for _ in range(1024)])
            tem = net(torch.from_numpy(xt).to(dev)).squeeze(1).cpu().numpy()
            bilan.setdefault("temoin", []).append((float(auc(tem, neg)), float((tem > seuil).mean()), float((tem > neg.max()).mean())))
            print(f"   témoin sans encre (forme de lettres remplie de papyrus vierge) : AUC {bilan['temoin'][-1][0]:.3f} ; détecté : {100 * bilan['temoin'][-1][1]:.0f} %")
        cands += [(v[0] - seuil,) + v[1:] for v in fen]                     # score ramené au seuil du réseau qui a noté
        torch.save(net.state_dict(), B / f"detecteur_pale_{pli}.pt")
    cands.sort(reverse=True)
    sc = np.array([c[0] for c in cands])
    print(f"balayage : {len(sc)} fenêtres notées par un réseau qui ne les a jamais vues ; au-dessus du seuil : {(sc > 0).sum()} (attendu par hasard : {len(sc) / 1000:.0f})")
    garde = []
    for c in cands:                                   # on écarte les fenêtres qui se chevauchent
        if all(abs(c[4] - g[4]) > 1.2 or abs(c[5] - g[5]) > 1.2 for g in garde):
            garde.append(c)
        if len(garde) == 24:
            break
    (B / "candidats_pales.json").write_text(json.dumps(dict(seuils=seuils, bilan=bilan, candidats=garde), indent=1))
    np.save(B / "scores_fenetres.npy", np.array([(c[0], c[4], c[5]) for c in cands], np.float32))
    print("  score au-dessus du seuil   x mm   y mm   tuile")
    for c in garde[:12]:
        print("  %6.2f                  %6.1f %6.1f   %s" % (c[0], c[4], c[5], c[1]))


if __name__ == "__main__":
    main()
