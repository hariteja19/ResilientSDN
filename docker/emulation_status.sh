#!/bin/sh
# Is an experiment campaign running in WSL? (run as root)
uptime
ps -eo etime,args | grep -E 'evaluation[.]experiment|run_wan_[e]xt|ryu-[m]anager' | cut -c1-110 || echo "no experiment process"
tail -2 /root/wan_ext.log 2>/dev/null
