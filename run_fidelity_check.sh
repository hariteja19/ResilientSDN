#!/bin/sh
# Emulation fidelity check (revision 4): re-run one cell of the 1 October campaign on the current
# WSL setup (Ubuntu 24.04, iperf3 3.16) and compare. October means (5 runs): NSFNET S3
# static 0.00% loss / 56 s above / 0 reroutes; simple reactive 0.00 / 23 / 39; ResilientSDN 0.00 / 18 / 2.
# Run only on an otherwise idle machine (after run_wan_ext.sh).
cd "$(dirname "$0")"
mkdir -p results/fidelity
./run.sh python3 -m evaluation.experiment --out results/fidelity --topologies nsfnet --scenarios S3 \
    --runs 3 | tee -a results/fidelity/campaign.log
