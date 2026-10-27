# bash validation/nl_groups/nl_groups_driver.sh <detector> <groups, e.g. 3,10,20> <uncal> ...
# Per segment: nl_groups_stage1.py on 3 chunks of integrations (one at a time), the self
# correction (nl_groups_self.py), stage 2 of every variant (two at a time), then the
# intermediates are removed; spec_<variant>.npz and self_coefficients.npz are kept in
# validation/runs/nl-groups/<detector>/seg<NNN>/.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
P=${PYTHON:-python}
detector=$1; groups=$2; shift 2
logs=validation/runs/nl-groups/logs
mkdir -p $logs
for uncal in "$@"; do
  s=$(basename $uncal | sed -E 's/.*-seg([0-9]+)_.*/\1/')
  out=validation/runs/nl-groups/$detector/seg$s
  if [ -f $out/complete ]; then continue; fi
  mkdir -p $out
  n=$($P -c "from astropy.io import fits; print(fits.getval('$uncal', 'NAXIS4', 1))")
  for j in 0 1 2; do
    a=$(( n * j / 3 )); b=$(( n * (j + 1) / 3 ))
    if [ -f $out/template_$(printf %05d $a).fits ]; then continue; fi
    $P "$HERE/nl_groups_stage1.py" $uncal $out $a $b $groups > $logs/stage1_${detector}_seg${s}_$a.log 2>&1
    echo "$detector seg$s stage 1 $a-$b: $(tail -1 $logs/stage1_${detector}_seg${s}_$a.log)"
  done
  if [ ! -f $out/self_coefficients.npz ]; then
    $P "$HERE/nl_groups_self.py" $out $groups > $logs/self_${detector}_seg$s.log 2>&1
    echo "$detector seg$s self: $(grep fits: $logs/self_${detector}_seg$s.log) $(tail -1 $logs/self_${detector}_seg$s.log)"
  fi
  variants=""
  for v in crds self; do for g in ${groups//,/ }; do variants="$variants ${v}_n$g"; done; done
  pids=""
  k=0
  for v in $variants; do
    if [ ! -f $out/spec_$v.npz ]; then
      $P "$HERE/nl_groups_stage2.py" $out $v $out/spec_$v.npz > $logs/stage2_${detector}_seg${s}_$v.log 2>&1 &
      pids="$pids $!"; k=$((k + 1))
      if [ $k -ge 2 ]; then wait $pids; pids=""; k=0; fi
    fi
  done
  wait $pids
  for v in $variants; do test -f $out/spec_$v.npz || { echo "missing $out/spec_$v.npz"; exit 1; }; done
  echo "$detector seg$s stage 2 done"
  rm -f $out/rawdark_*.npy $out/groupdq_*.npy $out/crds_n*_*.fits $out/self_n*_*.fits \
        $out/template_*.fits
  touch $out/complete
done
