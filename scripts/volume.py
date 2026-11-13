"""Accès aux données du bucket public, avec cache local dans data/.

- lire_region() : lit un pavé (z, y, x) d'un volume OME-Zarr en ne téléchargeant que les
  blocs nécessaires. Les blocs sont rangés comme par telecharger_volume.py, donc les deux
  outils partagent le même cache.
- recuperer() : télécharge un fichier du bucket dans data/ s'il n'y est pas déjà.
"""
import json
import os
import pathlib
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RACINE = pathlib.Path(__file__).resolve().parent.parent
# Le cache et les sorties vont dans <dépôt>/data ; LABO_DATA permet de pointer vers un autre dossier (cache existant).
DATA = pathlib.Path(os.environ["LABO_DATA"]).resolve() if os.environ.get("LABO_DATA") else RACINE / "data"
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"


def _get(url, tentatives=4):
    for essai in range(tentatives):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if essai == tentatives - 1:
                raise
        except Exception:
            if essai == tentatives - 1:
                raise


def recuperer(chemin, base=BUCKET):
    """Télécharge base+chemin vers data/chemin (une seule fois) et renvoie le chemin local."""
    local = DATA / chemin
    if not local.exists():
        contenu = _get(base + chemin)
        if contenu is None:
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".part")
        tmp.write_bytes(contenu)
        tmp.rename(local)
    return local


def lire_region(chemin_zarr, niveau, z, y, x, base=BUCKET, fils=16, cache=True, filtre=None):
    """Renvoie le pavé [z0:z1, y0:y1, x0:x1] du niveau demandé, en uint8.

    chemin_zarr : chemin dans le bucket, ex. "PHercParis4/volumes/....zarr/"
    z, y, x     : couples (début, fin) en voxels du niveau demandé.
    cache=False : les blocs sont lus en mémoire sans être écrits dans data/ (pour les gros
                  balayages, où le cache remplirait le disque).
    filtre      : ensemble d'indices de blocs (i, j, k) à lire ; les autres restent à zéro.
    """
    chemin_zarr = chemin_zarr.rstrip("/") + "/"
    za = json.loads(recuperer(f"{chemin_zarr}{niveau}/.zarray", base).read_text())
    codec = None
    if za.get("compressor") is not None:
        import numcodecs
        codec = numcodecs.get_codec(za["compressor"])
    forme, c = za["shape"], za["chunks"]
    sep = za.get("dimension_separator", ".")
    bornes = [(max(0, a), min(f, b)) for (a, b), f in zip((z, y, x), forme)]
    sortie = np.zeros([b - a for a, b in bornes], dtype=np.dtype(za["dtype"]))
    indices = [(i, j, k)
               for i in range(bornes[0][0] // c[0], (bornes[0][1] - 1) // c[0] + 1)
               for j in range(bornes[1][0] // c[1], (bornes[1][1] - 1) // c[1] + 1)
               for k in range(bornes[2][0] // c[2], (bornes[2][1] - 1) // c[2] + 1)]

    def decoder(octets):
        if codec is not None:
            octets = codec.decode(octets)
        return np.frombuffer(octets, dtype=sortie.dtype).reshape(c)

    def charger(idx):
        cle = sep.join(map(str, idx))
        if not cache:
            contenu = _get(f"{base}{chemin_zarr}{niveau}/{cle}")
            return None if contenu is None else decoder(contenu)
        local = DATA / chemin_zarr / str(niveau) / cle
        vide = local.with_name(local.name + ".vide")
        if vide.exists():
            return None
        if not local.exists():
            contenu = _get(f"{base}{chemin_zarr}{niveau}/{cle}")
            local.parent.mkdir(parents=True, exist_ok=True)
            if contenu is None:
                vide.touch()  # bloc absent du bucket = zone vide du masque
                return None
            tmp = local.with_name(f"{local.name}.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_bytes(contenu)
            os.replace(tmp, local)
            return decoder(contenu)
        return decoder(local.read_bytes())

    def traiter(idx):
        bloc = charger(idx)
        if bloc is None:
            return
        src, dst = [], []
        for d in range(3):
            debut = idx[d] * c[d]
            a = max(bornes[d][0], debut)
            b = min(bornes[d][1], debut + c[d])
            src.append(slice(a - debut, b - debut))
            dst.append(slice(a - bornes[d][0], b - bornes[d][0]))
        sortie[tuple(dst)] = bloc[tuple(src)]

    if filtre is not None:
        indices = [i for i in indices if i in filtre]
    with ThreadPoolExecutor(fils) as ex:
        list(ex.map(traiter, indices))
    return sortie


def lire_tif_region(chemin, y, x):
    """Lit le rectangle [y0:y1, x0:x1] d'un grand TIFF tuilé sans le charger en entier."""
    import tifffile
    (y0, y1), (x0, x1) = y, x
    with tifffile.TiffFile(chemin) as tif:
        page = tif.pages[0]
        haut, larg = page.shape[:2]
        th, tw = page.tilelength, page.tilewidth
        y0, y1, x0, x1 = max(0, y0), min(haut, y1), max(0, x0), min(larg, x1)
        sortie = np.zeros((y1 - y0, x1 - x0), dtype=page.dtype)
        par_ligne = -(-larg // tw)
        fh = tif.filehandle
        for ty in range(y0 // th, (y1 - 1) // th + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                i = ty * par_ligne + tx
                if page.databytecounts[i] == 0:
                    continue
                fh.seek(page.dataoffsets[i])
                tuile = page.decode(fh.read(page.databytecounts[i]), i)[0]
                tuile = np.squeeze(tuile)
                a, b = max(y0, ty * th), min(y1, ty * th + th)
                c, d = max(x0, tx * tw), min(x1, tx * tw + tw)
                sortie[a - y0:b - y0, c - x0:d - x0] = tuile[a - ty * th:b - ty * th, c - tx * tw:d - tx * tw]
    return sortie
