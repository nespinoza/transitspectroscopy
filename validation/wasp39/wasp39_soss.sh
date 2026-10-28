# bash validation/wasp39/wasp39_soss.sh: WASP-39 b NIRISS/SOSS (ERS 1366 obs 1), in order:
# self-correction fit on all chunks, CRDS stage 1 + stage 2, self stage 1 + stage 2.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
data=validation/data/wasp39_soss
out=validation/runs/wasp39/soss
logs=validation/runs/wasp39/logs
mkdir -p $out $logs
chunks() {  # first,last of the 3 chunks of a segment
  n=$($P -c "from astropy.io import fits; print(fits.getval('$1', 'NAXIS4', 1))")
  for j in 0 1 2; do echo "$(( n * j / 3 )) $(( n * (j + 1) / 3 ))"; done
}
uncals=$(ls $data/jw01366001001_04101_00001-seg00?_nis_uncal.fits)
stage1() {  # variant [coefficients]
  for u in $uncals; do
    s=$(basename $u | sed -E 's/.*-seg([0-9]+)_.*/\1/')
    chunks $u | while read a b; do
      tag=$(printf %05d $a)
      result=$out/seg$s/$1_$tag.fits; [ "$1" = fit ] && result=$out/seg$s/selffit_$tag.npz
      [ -f $result ] && continue
      $P "$HERE/wasp39_stage1.py" $u $out/seg$s $a $b $1 $2 > $logs/soss_$1_seg${s}_$a.log 2>&1
      echo "soss $1 seg$s $a-$b: $(tail -1 $logs/soss_$1_seg${s}_$a.log)"
    done
  done
}
stage2() {  # variant: all chunks at once (one trace and background for the exposure)
  [ -f $out/spec_$1_order1.npz ] && return 0
  $P "$HERE/wasp39_soss_stage2.py" $out $1 $out/spec_$1 > $logs/soss_stage2_$1.log 2>&1
  echo "soss stage 2 $1: $(tail -1 $logs/soss_stage2_$1.log)"
}
stage1 fit
[ -f $out/self_linearity.npz ] || $P "$HERE/wasp39_combine_fit.py" $out $out/self_linearity.npz | tee $logs/soss_combine.log
stage1 crds
stage2 crds
stage1 self $out/self_linearity.npz
stage2 self
echo "SOSS DONE"
