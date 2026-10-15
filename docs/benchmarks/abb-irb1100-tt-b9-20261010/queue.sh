#!/bin/sh
# The study's runs (docs/benchmarks/abb-irb1100-tt-b9-20261010.md), two containers at a time (shared host):
#   RECV=<tt-receive-study checkout> ./queue.sh       then  python3 summarize.py
cd "$(dirname "$0")"
export RECV=${RECV:?}
( NAME=abb-b9-qA NPROC=8 ./run.sh oracle_abb.py or_base
  ORACLE=or_base READY=home,near_fh,near_bh NAME=abb-b9-qA NPROC=8 ./run.sh fullsim_abb.py fs_base
  N_RALLIES=24 SEEDS=1 NAME=abb-b9-qA NPROC=8 ./run.sh rally_abb.py rally_base
  LEAD=0.084 REACT=0.04 NAME=abb-b9-qA NPROC=8 ./run.sh oracle_abb.py or_base_lat ) > results/queue_A.log 2>&1 &
( ABB_PADDLE=0.263 NAME=abb-b9-qB NPROC=8 ./run.sh oracle_abb.py or_ext
  ABB_PADDLE=0.263 ORACLE=or_ext READY=home,near_fh,near_bh NAME=abb-b9-qB NPROC=8 ./run.sh fullsim_abb.py fs_ext
  ABB_PADDLE=0.263 LEAD=0.084 REACT=0.04 NAME=abb-b9-qB NPROC=8 ./run.sh oracle_abb.py or_ext_lat
  ABB_PADDLE=0.263 LEAD=0.084 REACT=0.04 ORACLE=or_ext_lat READY=home,near_fh,near_bh NAME=abb-b9-qB NPROC=8 ./run.sh fullsim_abb.py fs_ext_lat ) > results/queue_B.log 2>&1 &
wait
# afterwards (each in its own container): the reach scans and the serve capability
#   ./run.sh reach_scan.py reach_base;  ABB_PADDLE=0.263 ./run.sh reach_scan.py reach_ext
#   ABB_DROP=0.2 ./run.sh reach_scan.py reach_drop20;  ABB_DROP=0.2 ABB_PADDLE=0.263 ./run.sh reach_scan.py reach_extdrop20
#   ./run.sh servecap_abb.py servecap_abb           (NPROC=8 for the scans)
