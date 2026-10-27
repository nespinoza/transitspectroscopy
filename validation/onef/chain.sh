# bash validation/onef/chain.sh <name> <uncal> ...: the group 1/f inputs of every segment, in 3
# chunks of integrations, one at a time (Stage 1 of a whole segment does not fit in 8 GB).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
name=$1; shift
mkdir -p validation/runs/onef-full/logs
for uncal in "$@"; do
  seg=$(basename $uncal | sed -E 's/.*-seg([0-9]+)_.*/\1/')
  out=validation/runs/onef-full/$name/seg$seg
  n=$($P -c "from astropy.io import fits; print(fits.getval('$uncal', 'NAXIS4', 1))")
  for j in 0 1 2; do
    a=$(( n * j / 3 )); b=$(( n * (j + 1) / 3 ))
    if [ -f $out/geometry_$(printf %05d $a).npz ]; then continue; fi
    log=validation/runs/onef-full/logs/extract_${name}_seg${seg}_$a.log
    $P "$HERE/onef_extract.py" $uncal $out $a $b > $log 2>&1
    echo "seg$seg $a-$b: $(tail -1 $log)"
  done
done
