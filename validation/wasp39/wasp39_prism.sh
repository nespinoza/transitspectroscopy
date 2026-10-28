# bash validation/wasp39/wasp39_prism.sh: WASP-39 b NIRSpec PRISM (ERS 1366 obs 4), in order:
# self-correction fit on all chunks, CRDS stage 1 + stage 2, self stage 1 + stage 2.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
data=validation/data/wasp39_prism
out=validation/runs/wasp39/prism
logs=validation/runs/wasp39/logs
mkdir -p $out $logs
chunks() {  # first,last of the 3 chunks of a segment
  n=$($P -c "from astropy.io import fits; print(fits.getval('$1', 'NAXIS4', 1))")
  for j in 0 1 2; do echo "$(( n * j / 3 )) $(( n * (j + 1) / 3 ))"; done
}
uncals=$(ls $data/jw01366004001_04101_00001-seg00?_nrs1_uncal.fits)
stage1() {  # variant [coefficients]
  for u in $uncals; do
    s=$(basename $u | sed -E 's/.*-seg([0-9]+)_.*/\1/')
    chunks $u | while read a b; do
      tag=$(printf %05d $a)
      result=$out/seg$s/$1_$tag.fits; [ "$1" = fit ] && result=$out/seg$s/selffit_$tag.npz
      [ -f $result ] && continue
      $P "$HERE/wasp39_stage1.py" $u $out/seg$s $a $b $1 $2 > $logs/prism_$1_seg${s}_$a.log 2>&1
      echo "prism $1 seg$s $a-$b: $(tail -1 $logs/prism_$1_seg${s}_$a.log)"
    done
  done
}
stage2() {  # variant
  reference=""  # one trace for all segments and variants: that of CRDS segment 1
  if [ "$1" != crds ]; then reference=$out/spec_crds_seg001; fi
  for u in $uncals; do
    s=$(basename $u | sed -E 's/.*-seg([0-9]+)_.*/\1/')
    firsts=$(chunks $u | awk '{printf "%s%d", (NR>1?",":""), $1}')
    spec=$out/spec_$1_seg$s.npz
    if [ ! -f $spec ]; then
      $P "$HERE/wasp39_stage2.py" $out/seg$s $1 $firsts $spec $reference > $logs/prism_stage2_$1_seg$s.log 2>&1
      echo "prism stage 2 $1 seg$s: $(tail -1 $logs/prism_stage2_$1_seg$s.log)"
    fi
    if [ -z "$reference" ]; then reference=$out/spec_crds_seg001; fi
  done
}
stage1 fit
[ -f $out/self_linearity.npz ] || $P "$HERE/wasp39_combine_fit.py" $out $out/self_linearity.npz | tee $logs/prism_combine.log
stage1 crds
stage2 crds
stage1 self $out/self_linearity.npz
stage2 self
echo "PRISM DONE"
