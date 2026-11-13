#!/usr/bin/env python3
"""Recense les segments déjà déroulés et publiés pour chaque rouleau, avec ce qu'il faut pour y passer un modèle d'encre.

Pour chaque segment du fichier public metadata.min.json : volumes de surface publiés (résolution, taille),
cartes d'encre déjà publiées, et éligibilité du rouleau aux prix (d'après catalogue/statut_lecture.csv).
Ne télécharge que des en-têtes (.zarray), aucun volume.

  .venv/bin/python -I scripts/recenser_segments.py        -> catalogue/segments.csv et un résumé
"""
import csv
import json
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import construire_catalogue as CC  # noqa: E402
import volume as V  # noqa: E402

RACINE = pathlib.Path(__file__).resolve().parent.parent


def forme_zarr(chemin):
    """(couches, hauteur, largeur) du niveau 0 d'un volume de surface, ou None."""
    for sous in ("0/.zarray", ".zarray"):
        x = V._get(V.BUCKET + chemin.rstrip("/") + "/" + sous)
        if x:
            try:
                return tuple(json.loads(x)["shape"])
            except Exception:
                pass
    return None


def main():
    statut = {r["sample_id"]: r for r in csv.DictReader(open(RACINE / "catalogue/statut_lecture.csv", encoding="utf-8"))}
    meta = CC.lire_json(CC.BUCKET + "metadata.min.json")["samples"]
    lignes, taches = [], []
    for sid, e in meta.items():
        st = statut.get(sid, {})
        for nom, seg in (e.get("segments") or {}).items():
            sv = [d for d in seg.get("data", []) if d.get("type") == "layers-zarr"]
            autres = sorted({d.get("type") for d in seg.get("data", [])} - {"layers-zarr", "tifxyz", "obj", "obj-flattened", "obj-normalized", "tifxyz-normalized", "tifxyz-transformed"})
            for d in sv or [None]:
                chemin = d["origins"][0]["path"] if d else ""
                m = re.search(r"([\d.]+)um", chemin.split("surface-volumes/")[-1]) if chemin else None
                ligne = dict(sample_id=sid, type=st.get("type", ""), statut=st.get("statut", ""), prix_ouverts=st.get("prix_ouverts", ""), segment=seg.get("long_id", nom),
                             volume_surface=chemin, pixel_um=float(m.group(1)) if m else "", couches="", hauteur_mm="", largeur_mm="", surface_cm2="", autres_donnees=" ".join(autres))
                lignes.append(ligne)
                if chemin:
                    taches.append(ligne)

    def remplir(l):
        f = forme_zarr(l["volume_surface"])
        if f and l["pixel_um"]:
            l["couches"] = f[0]; l["hauteur_mm"] = round(f[1] * l["pixel_um"] / 1000, 1); l["largeur_mm"] = round(f[2] * l["pixel_um"] / 1000, 1)
            l["surface_cm2"] = round(l["hauteur_mm"] * l["largeur_mm"] / 100, 1)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(remplir, taches))
    sortie = RACINE / "catalogue/segments.csv"
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys())); w.writeheader(); w.writerows(lignes)
    print(f"{len(lignes)} lignes écrites dans {sortie}")
    # résumé par rouleau
    par = {}
    for l in lignes:
        p = par.setdefault(l["sample_id"], dict(prix=l["prix_ouverts"], statut=l["statut"], type=l["type"], segs=set(), avec=set(), res=set(), cm2=0.0, autres=set()))
        p["segs"].add(l["segment"])
        if l["volume_surface"]:
            p["avec"].add(l["segment"]); p["res"].add(l["pixel_um"]); p["cm2"] += l["surface_cm2"] or 0
        p["autres"].update(l["autres_donnees"].split())
    print("\nrouleau        prix    statut       segments  avec volume de surface  résolutions (µm)      surface (cm²)  autres données publiées")
    for sid, p in sorted(par.items(), key=lambda t: (("FL" not in t[1]["prix"]), -len(t[1]["avec"]))):
        if p["type"] != "rouleau":
            continue
        print("%-14s %-7s %-12s %5d      %5d               %-20s %8.0f       %s" % (sid, p["prix"] or "-", p["statut"], len(p["segs"]), len(p["avec"]), ", ".join(str(r) for r in sorted(p["res"])), p["cm2"], " ".join(sorted(p["autres"]))[:60]))
    sans = [sid for sid, r in statut.items() if "FL" in r.get("prix_ouverts", "") and sid not in par]
    print("\néligibles à First Letters sans aucun segment publié :", ", ".join(sorted(sans)))


if __name__ == "__main__":
    main()
