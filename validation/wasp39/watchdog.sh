# bash validation/wasp39/watchdog.sh <log>: logs free memory, swap, free disk and the footprint
# of the nl_groups python jobs; kills them when swap > 8 GB, their footprint > 9 GB, free
# memory < 10 % or free disk < 3 GB.
LOG=$1
while true; do
  lvl=$(sysctl -n kern.memorystatus_level)
  swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
  disk=$(df -g / | awk 'NR==2 {print $4}')
  pids=$(pgrep -f 'python.*wasp3[9]_' | tr '\n' ',' | sed 's/,$//')
  foot=0
  if [ -n "$pids" ]; then
    foot=$(top -l 1 -pid ${pids//,/ -pid } -stats mem 2>/dev/null | tail -n +13 | awk '
      /G/ {gsub("G[+-]?",""); s+=$1*1024; next} /M/ {gsub("M[+-]?",""); s+=$1} END {print int(s)}')
  fi
  echo "$(date +%H:%M:%S) free=${lvl}% swap=${swap}M disk=${disk}G jobs_footprint=${foot}M" >> $LOG
  if [ "$swap" -gt 8000 ] || [ "${foot:-0}" -gt 9000 ] || [ "$lvl" -lt 10 ] || [ "$disk" -lt 3 ]; then
    echo "$(date +%H:%M:%S) LIMIT: killing wasp39 jobs" >> $LOG
    pkill -f 'python.*wasp3[9]_'
  fi
  sleep 15
done
