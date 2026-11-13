#!/bin/zsh
# ink_9um sur la fin de PHerc. Paris 4 : zone vierge après la dernière colonne, puis sous la dernière colonne.
cd "$(dirname "$0")/.." || exit 1      # racine du dépôt
mkdir -p data/nuit
.venv/bin/python -I scripts/ink9.py paris4 p4_vierge 0 167.4 281 408 0 >> data/nuit/ink9_vierge.log 2>&1
.venv/bin/python -I scripts/ink9.py paris4 p4_sous_colonne 44 167.4 205 281 0 >> data/nuit/ink9_sous_colonne.log 2>&1
touch data/nuit/INK9_FINI
