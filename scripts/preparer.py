#!/usr/bin/env python3
"""Prépare un dépôt neuf : vérifie le code et les poids des modèles, et télécharge dans data/ les petits
fichiers d'entrée que les autres scripts attendent (maillages des deux segments, matrices de passage).

Les volumes eux-mêmes ne sont jamais téléchargés en entier : chaque script lit dans le bucket public
les seuls blocs dont il a besoin (scripts/volume.py).

  .venv/bin/python -I scripts/preparer.py                      # dit ce qui manque, ne télécharge rien
  .venv/bin/python -I scripts/preparer.py --executer           # maillages (≈ 360 Mo) et matrices de passage
  .venv/bin/python -I scripts/preparer.py --executer --cartes  # + les deux cartes d'encre 2D publiées de la fin de Paris 4 (≈ 2 Go),
                                                               #   nécessaires à detecteur_carte.py canon | fin2d
"""
import argparse
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import volume as V  # noqa: E402

RACINE = V.RACINE
P4 = "PHercParis4/segments/"
MAILLAGES = [
    P4 + "20260701183124-w010-027/mesh/20260701183124-on-20260411134726-2.4um.tifxyz/",          # fin de Paris 4 (segment « 701 »)
    "PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi/mesh/20260422000000-on-20260102150214-2.399um.tifxyz/",   # titre de PHerc. 139
]
CARTES = [
    P4 + "20260701183124-w010-027/ink-detection/PHercParis4-20260701183124-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif",
    P4 + "20260623141924-w010-027/ink-detection/PHercParis4-20260623141924-1.129um-0.23m-78keV-volume-20260608103018-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif",
]
CODE = [
    ("externe/villa/vesuvius/src/vesuvius/ink_detection/paris4_3d_gambiarra/student.py",
     "git clone --filter=blob:none --branch ink-paris4-3d-gambiarra https://github.com/ScrollPrize/villa externe/villa"),
    ("externe/villa-ink9/ink-detection/koine_machines/inference/infer.py",
     "git clone --filter=blob:none --branch merge-ink-pipelines https://github.com/ScrollPrize/villa externe/villa-ink9"),
    ("modeles/ink_paris4_3d_4.8um_gambiarra/model.safetensors",
     "curl -L -o modeles/ink_paris4_3d_4.8um_gambiarra/model.safetensors https://huggingface.co/scrollprize/ink_paris4_3d_4.8um_gambiarra/resolve/main/model.safetensors"),
    ("modeles/ink_9um/hybrid_3d2d-seed42-step-075000.pth",
     "curl -L -o modeles/ink_9um/hybrid_3d2d-seed42-step-075000.pth https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth   (de même pour seed43)"),
]


def taille(chemin):
    req = urllib.request.Request(V.BUCKET + chemin, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def telecharger(chemin):
    """Télécharge par morceaux (les cartes pèsent 1 Go : pas question de les garder en mémoire)."""
    local = V.DATA / chemin
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    req = urllib.request.Request(V.BUCKET + chemin, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            bloc = r.read(1 << 22)
            if not bloc:
                break
            f.write(bloc)
    tmp.rename(local)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executer", action="store_true", help="télécharger ce qui manque (sans cette option : simple liste)")
    ap.add_argument("--cartes", action="store_true", help="ajouter les deux cartes d'encre 2D publiées (≈ 2 Go)")
    a = ap.parse_args()

    print("Code et poids des modèles (à installer à la main, voir README.md) :")
    for chemin, aide in CODE:
        present = (RACINE / chemin).exists()
        print(f"   {'présent ' if present else 'MANQUANT'}  {chemin}" + ("" if present else f"\n             -> {aide}"))

    fichiers = [m + f for m in MAILLAGES for f in ("meta.json", "x.tif", "y.tif", "z.tif")] + (CARTES if a.cartes else [])
    print(f"\nFichiers d'entrée dans {V.DATA} :")
    total = 0
    for chemin in fichiers:
        if (V.DATA / chemin).exists():
            print(f"   présent   {chemin}")
            continue
        n = taille(chemin); total += n
        print(f"   {'téléchargement' if a.executer else 'manquant'}  {n / 1e6:8.1f} Mo  {chemin}", flush=True)
        if a.executer:
            telecharger(chemin)

    matrices = [V.DATA / "PHercParis4" / n for n in ("transfo_24_vers_1129.npy", "transfo_78_vers_137.npy")]
    if all(m.exists() for m in matrices):
        print("   présent   matrices de passage entre scans (data/PHercParis4/transfo_*.npy)")
    elif a.executer:
        import subprocess
        subprocess.run([sys.executable, "-I", str(pathlib.Path(__file__).with_name("transfos.py"))], check=True)
    else:
        print("   manquant  matrices de passage entre scans -> scripts/transfos.py (quelques ko)")
    if not a.executer and total:
        print(f"\n{total / 1e6:.0f} Mo à télécharger ; relancer avec --executer.")
