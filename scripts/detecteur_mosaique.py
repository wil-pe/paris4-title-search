#!/usr/bin/env python3
"""Deuxième détecteur d'écriture pâle sur le balayage du scan 1,129 µm : fenêtres larges.

`detecteur_pale.py` regarde des fenêtres de 1,5 mm à l'intérieur de chaque tuile (un morceau de lettre).
Celui-ci assemble les tuiles en une seule carte, la réduit à 19,2 µm par pixel et regarde des fenêtres
de 4,9 mm (une lettre entière et ses voisines), ce qui convient mieux à un titre en grandes lettres.
Même principe : vrai papyrus vierge contre le même papyrus + traits de l'étalonnage (encre affaiblie),
lettres de test jamais vues, et chaque bande de 8 mm notée par un réseau qui ne l'a pas vue.

  .venv/bin/python -I scripts/detecteur_mosaique.py
"""
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mosaique_balayage as M  # noqa: E402

B = M.B
G = 2
MM = 0.0096 * G
W = 256
PAS_F = 64
BANDE = 8.0
ALPHAS = ("a060", "a040", "a025", "a015", "a000")
rng = np.random.default_rng(1)


def reduire(a):
    h, w = a.shape[0] // G, a.shape[1] // G
    return a[:h * G, :w * G].reshape(h, G, w, G).mean((1, 3))


def norm(a):
    return (np.clip(a, -8, 6) / 4).astype(np.float32)


def xmm(c):
    return (M.X0 + np.asarray(c) * 4 * G) * 0.0024


def parite(c):
    return (np.floor(xmm(np.asarray(c) + W / 2) / BANDE) % 2).astype(int)


def reseau():
    def bloc(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(4, o), nn.ReLU())
    return nn.Sequential(bloc(1, 16), bloc(16, 32), bloc(32, 64), bloc(64, 64), bloc(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))


def traits(mos, alpha, x0, x1, echelle):
    """Bande de lettres affaiblies (hauteur 3 mm) posée dans une fenêtre W x W ; -30 hors des traits."""
    enc = mos["encre"]; t = mos[alpha]
    l = int(W / echelle)
    for _ in range(50):
        c = rng.integers(x0, x1 - l + 1)
        if enc[:, c:c + l].mean() > 0.06:
            break
    v = np.where(ndi.binary_dilation(enc[:, c:c + l], iterations=2), t[:, c:c + l], -30).astype(np.float32)
    if echelle != 1.0:
        v = ndi.zoom(v, echelle, order=1)[:, :W]
    h = min(v.shape[0], W); out = np.full((W, W), -30, np.float32)
    r = rng.integers(0, W - h + 1)
    out[r:r + h, :v.shape[1]] = v[:h]
    return out[:, ::-1] if rng.random() < 0.5 else out


def lot(carte, coins, mos, x0, x1, n, alphas=ALPHAS):
    x = np.empty((n, 1, W, W), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        r, c = coins[rng.integers(len(coins))]
        f = carte[r:r + W, c:c + W]
        if rng.random() < 0.5:
            f = f[:, ::-1]
        if rng.random() < 0.5:
            f = f[::-1]
        if i % 2:
            f = np.maximum(f, traits(mos, alphas[rng.integers(len(alphas))], x0, x1, float(rng.choice([1.0, 1.0, 1.3, 1.6])))); y[i] = 1
        x[i, 0] = norm(f)
    return torch.from_numpy(x), torch.from_numpy(y)


def noter(net, carte, coins, taille=256):
    out = np.empty(len(coins), np.float32)
    with torch.inference_mode():
        for i in range(0, len(coins), taille):
            x = np.stack([norm(carte[r:r + W, c:c + W])[None] for r, c in coins[i:i + taille]])
            out[i:i + taille] = net(torch.from_numpy(x)).squeeze(1).numpy()
    return out


def planche(carte, cands, n=16):
    cote, cols = 312, 4; js = cands[:n]; lignes = -(-len(js) // cols)
    im = Image.new("RGB", (cols * (cote + 8), lignes * (cote + 24)), (20, 20, 20)); d = ImageDraw.Draw(im)
    for i, k in enumerate(js):
        r = int((k["y_mm"] / 0.0024 - M.Y0) / 4 / G); c = int((k["x_mm"] / 0.0024 - M.X0) / 4 / G); px, py = (i % cols) * (cote + 8), (i // cols) * (cote + 24)
        v = np.full((cote, cote), -30, np.float32)
        r0, r1, c0, c1 = max(r - cote // 2, 0), min(r + cote // 2, carte.shape[0]), max(c - cote // 2, 0), min(c + cote // 2, carte.shape[1])
        v[r0 - (r - cote // 2):r1 - (r - cote // 2), c0 - (c - cote // 2):c1 - (c - cote // 2)] = carte[r0:r1, c0:c1]
        im.paste(Image.fromarray(M.gris(v)), (px, py + 22))
        d.text((px + 4, py + 5), f"#{i + 1} score {k['score']:+.1f}  x {k['x_mm']:.1f}  y {k['y_mm']:.1f} mm", fill=(255, 220, 120))
    im.save(M.APERCUS / "paris4_candidats_mosaique.png")


def main():
    torch.set_num_threads(4)
    m, ok, fait = M.mosaique()
    carte = reduire(m).astype(np.float32); bon = reduire((ok & fait).astype(np.float32)) > 0.99
    z = dict(np.load(B / "etalon_texte_mosaique.npz"))
    mos = {a: reduire(z[f"s11_{a}_p"].astype(np.float32)) for a in ALPHAS}
    mos["encre"] = reduire((z["s11_a100_p"] > 0).astype(np.float32)) > 0.5
    largeur = mos["encre"].shape[1]; coupe = int(largeur * 2 / 3)
    part = ndi.uniform_filter(bon.astype(np.float32), W, mode="constant")[W // 2::PAS_F, W // 2::PAS_F]
    rr, cc = np.nonzero(part >= 0.97); g = (rr * PAS_F + W <= carte.shape[0]) & (cc * PAS_F + W <= carte.shape[1])
    grille = np.stack([rr[g] * PAS_F, cc[g] * PAS_F], 1)
    print(f"carte {carte.shape} à {1000 * MM:.1f} µm ; {len(grille)} fenêtres de {W * MM:.1f} mm entièrement calculées", flush=True)
    scores = np.full(len(grille), np.nan, np.float32); bilan = {}; seuils = []
    for pli in (0, 1):
        app = grille[parite(grille[:, 1]) == pli]; test = grille[parite(grille[:, 1]) != pli]
        net = reseau(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2); perte = nn.BCEWithLogitsLoss()
        net.train()
        for pas in range(1200):
            x, y = lot(carte, app, mos, 0, coupe, 48)
            for g in opt.param_groups:
                g["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * pas / 1200))
            opt.zero_grad(); p = perte(net(x).squeeze(1), y); p.backward(); opt.step()
        net.eval()
        neg = noter(net, carte, test); seuil = float(np.quantile(neg, 0.999)); seuils.append(seuil)
        vus = noter(net, carte, app[::4])
        print(f"réseau {pli} : vierge jamais vu ({len(test)} fenêtres) : médiane {np.median(neg):.1f}, 99 % {np.quantile(neg, 0.99):.1f}, 99,9 % {seuil:.1f}, max {neg.max():.1f} | vierge vu à l'entraînement : médiane {np.median(vus):.1f}")
        with torch.inference_mode():
            for a in ALPHAS:
                pos = np.concatenate([net(lot(carte, test, mos, coupe, largeur, 256, (a,))[0][1::2]).squeeze(1).numpy() for _ in range(3)])
                bilan.setdefault(a, []).append((float((pos > seuil).mean()), float((pos > neg.max()).mean())))
                print(f"   encre à {int(a[1:]):3d} % : détectée au seuil 1/1000 : {100 * bilan[a][-1][0]:.0f} % ; au-dessus de TOUT le vierge : {100 * bilan[a][-1][1]:.0f} %", flush=True)
        # test direct, sans montage : bandes brutes de l'étalonnage (3 mm de haut, dernier tiers de la ligne),
        # comparées à des bandes de papyrus vierge de même taille prises dans les fenêtres jamais vues
        with torch.inference_mode():
            hb = mos["encre"].shape[0]
            vb = np.concatenate([net(torch.from_numpy(np.stack([norm(carte[r + o:r + o + hb, c:c + W])[None] for r, c in test[i:i + 256]]))).squeeze(1).numpy()
                                 for o in (0, W - hb) for i in range(0, len(test), 256)])
            sb = float(np.quantile(vb, 0.999))
            for a in ("a100",) + ALPHAS:
                brut = reduire(z[f"s11_{a}_p"].astype(np.float32))
                cs = list(range(coupe, largeur - W + 1, 16))
                sc = net(torch.from_numpy(np.stack([norm(brut[:, c:c + W])[None] for c in cs]))).squeeze(1).numpy()
                bilan.setdefault("direct_" + a, []).append((float((sc > sb).mean()), float((sc > vb.max()).mean()), float(np.median(sc))))
                print(f"   test direct, encre à {int(a[1:]):3d} % ({len(cs)} bandes) : au-dessus du seuil 1/1000 : {100 * (sc > sb).mean():.0f} % ; au-dessus de TOUT le vierge : {100 * (sc > vb.max()).mean():.0f} % ; score médian {np.median(sc):.1f} (vierge : médiane {np.median(vb):.1f}, max {vb.max():.1f})", flush=True)
        mk = parite(grille[:, 1]) != pli
        scores[mk] = neg - seuil
        torch.save(net.state_dict(), B / f"detecteur_mosaique_{pli}.pt")
    print(f"fenêtres notées : {len(grille)} ; au-dessus du seuil : {(scores > 0).sum()} (attendu par hasard : {len(grille) / 1000:.0f})")
    cands = []
    for i in np.argsort(-scores):
        r, c = grille[i]; k = dict(score=float(scores[i]), x_mm=float(xmm(c + W / 2)), y_mm=float((M.Y0 + (r + W / 2) * 4 * G) * 0.0024))
        if all(abs(k["x_mm"] - q["x_mm"]) > 3 or abs(k["y_mm"] - q["y_mm"]) > 3 for q in cands):
            cands.append(k)
        if len(cands) == 24:
            break
    (B / "candidats_mosaique.json").write_text(json.dumps(dict(seuils=seuils, bilan=bilan, candidats=cands), indent=1))
    print("   score au-dessus du seuil   x mm   y mm")
    for k in cands[:12]:
        print("   %+6.2f                 %6.1f %6.1f" % (k["score"], k["x_mm"], k["y_mm"]))
    planche(carte, cands)


if __name__ == "__main__":
    main()
