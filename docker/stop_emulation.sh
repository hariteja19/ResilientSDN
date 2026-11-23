#!/bin/sh
# Stop a running experiment campaign and clean Mininet up (run as root in WSL).
for pat in 'run_wan_[e]xt' 'evaluation[.]experiment' 'ryu-[m]anager' '[i]perf3'; do
    for pid in $(ps -eo pid,args | grep -E "$pat" | awk '{print $1}'); do
        [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
    done
done
sleep 2
mn -c >/dev/null 2>&1
echo "still running:"
ps -eo pid,args | grep -E 'evaluation[.]experiment|ryu-[m]anager|[i]perf3|run_wan_[e]xt' || echo "  nothing"
