#!/bin/zsh
# ink_9um sur les six segments publiés de PHerc. 800 (éligible à First Letters), à trois profondeurs.
cd "$(dirname "$0")/.." || exit 1      # racine du dépôt
mkdir -p data/nuit
.venv/bin/python -I scripts/ink9.py natif p800_s1 PHerc0800/segments/20251028222030-auto_grown_20251028222030940/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
.venv/bin/python -I scripts/ink9.py natif p800_s2 PHerc0800/segments/20251028225813-auto_grown_20251028225813045/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
.venv/bin/python -I scripts/ink9.py natif p800_s3 PHerc0800/segments/20251028213516-auto_grown_20251028213516907/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
.venv/bin/python -I scripts/ink9.py natif p800_s4 PHerc0800/segments/20251028220042-auto_grown_20251028220042762/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
.venv/bin/python -I scripts/ink9.py natif p800_s5 PHerc0800/segments/20251028220955-auto_grown_20251028220955262/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
.venv/bin/python -I scripts/ink9.py natif p800_s6 PHerc0800/segments/20251029010146-auto_grown_20251029010146642/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/ -5,0,5 >> data/nuit/ink9_p800.log 2>&1
touch data/nuit/P800_FINI
