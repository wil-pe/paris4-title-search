#!/bin/zsh
# Balayage du scan 1,129 µm sur la zone vierge après la dernière colonne de PHerc. Paris 4.
# Chaque processus repart à neuf toutes les 10 tuiles (code de sortie 75) pour ne pas accumuler de mémoire.
cd "$(dirname "$0")/.." || exit 1      # racine du dépôt
mkdir -p data/nuit
J=data/nuit/journal.txt
N=${1:-2}
boucle() {   # $1 = part (i/n), $2 = journal
  for k in {1..200}; do
    .venv/bin/python -I scripts/balayage.py scan fin_1129 20000 44320 118750 144350 11 $1 10 >> $2 2>&1
    [ $? -eq 0 ] && break
    sleep 3
  done
}
echo "=== balayage du scan 1,129 µm ($N processus), x 285-345 mm, y 48-106 mm $(date '+%H:%M:%S')" >> $J
for i in $(seq 0 $((N - 1))); do
  boucle $i/$N data/nuit/journal_$i.txt &
done
wait
echo "=== fin $(date '+%H:%M:%S')" >> $J
touch data/nuit/FINI
