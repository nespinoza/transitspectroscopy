# bash validation/wasp39/wasp39_prism_masked.sh <variant> [<variant> ...]: PRISM stage 2 and
# spectrum with the saturated core masked (wasp39_saturation_mask.py), for the masks 'core'
# (pixels saturating in > 0.1% of the integrations) and 'grow1' (also +-1 row), after
# wasp39_prism.sh; the spectra are differenced with the unmasked ones of the same variant.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
out=validation/runs/wasp39/prism
logs=validation/runs/wasp39/logs
E=$($P -c "import numpy as np; print(','.join(f'{e:.2f}' for e in np.arange(0.60, 5.3401, 0.04)))")
[ -f $out/saturation_mask_core.npy ] || \
  $P "$HERE/wasp39_saturation_mask.py" $out crds $out/saturation_mask_core.npy 0 | tee $logs/mask_core.log
[ -f $out/saturation_mask_grow1.npy ] || \
  $P "$HERE/wasp39_saturation_mask.py" $out crds $out/saturation_mask_grow1.npy 1 | tee $logs/mask_grow1.log
for v in "$@"; do
  for m in core grow1; do
    for s in 001 002 003 004; do
      n=$(ls $out/seg$s/${v}_?????.fits | wc -l | tr -d ' ')
      firsts=$(ls $out/seg$s/${v}_?????.fits | sed -E 's/.*_0*([0-9]+)\.fits/\1/' | sort -n | paste -sd, -)
      spec=$out/spec_${v}_${m}_seg$s.npz
      if [ ! -f $spec ]; then
        $P "$HERE/wasp39_stage2.py" $out/seg$s $v $firsts $spec $out/spec_crds_seg001 \
          $out/saturation_mask_$m.npy > $logs/prism_stage2_${v}_${m}_seg$s.log 2>&1
        echo "prism stage 2 ${v} ${m} seg$s ($n chunks): $(tail -1 $logs/prism_stage2_${v}_${m}_seg$s.log)"
      fi
    done
    if [ ! -f $out/spectrum_${v}_${m}.npz ]; then
      $P "$HERE/wasp39_spectrum.py" prism $out/spectrum_${v}_${m}.npz $out/spec_${v}_${m}_seg00{1,2,3,4}.npz \
        --edges $E --skip 1000 --reference $out/spec_${v}_seg00{1,2,3,4}.npz 2>&1 | tee $logs/prism_spectrum_${v}_${m}.log
    fi
  done
done
echo "MASKED DONE"
