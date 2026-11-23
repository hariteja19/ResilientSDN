#!/bin/sh
# Revision 4: the scenarios the first WAN campaign left out (S6, S7, S8, S10) on NSFNET, Internet2 and
# GEANT, same protocol as run_full.sh (3 systems, 5 runs). Short scenarios first; S6 (600 s attacks)
# last. Safe to stop and restart: finished runs are skipped.
cd "$(dirname "$0")"
mkdir -p results/wan_ext
./run.sh python3 -m evaluation.experiment --out results/wan_ext --topologies nsfnet internet2 geant \
    --scenarios S7 S8 S10 --runs 5 | tee -a results/wan_ext/campaign.log
./run.sh python3 -m evaluation.experiment --out results/wan_ext --topologies nsfnet internet2 geant \
    --scenarios S6 --runs 5 | tee -a results/wan_ext/campaign.log
echo "WAN extension finished"
