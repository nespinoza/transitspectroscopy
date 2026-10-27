# bash validation/onef/sweep_driver.sh <name> <effinttm> <inners> <widths> <apertures> <uncal> ...
# Per segment: the chunked inputs (chain.sh), the 1/f pixel sweep (onef_sweep.py), then the
# inputs are removed (tens of GB per exposure); the sweep_seg*.npz results are kept.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
name=$1; tint=$2; inners=$3; widths=$4; aps=$5; shift 5
mkdir -p validation/runs/onef-full/logs
for uncal in "$@"; do
  s=$(basename $uncal | sed -E 's/.*-seg([0-9]+)_.*/\1/')
  out=validation/runs/onef-full/$name/seg$s
  res=validation/runs/onef-full/$name/sweep_seg$s.npz
  if [ -f $res ]; then continue; fi
  bash "$HERE/chain.sh" $name $uncal
  log=validation/runs/onef-full/logs/sweep_${name}_seg$s.log
  $P "$HERE/onef_sweep.py" $out $tint $inners $widths $aps $res > $log 2>&1
  echo "$name seg$s sweep: $(tail -1 $log)"
  rm -f $out/delta_*.npy $out/masks_*.npy $out/rates_*.npy $out/geometry_*.npz \
        $out/reference_*.npy $out/mode_*.txt
done
