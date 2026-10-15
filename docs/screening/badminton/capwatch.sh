#!/bin/sh
# MAXC = 2 while a model match (badminton-match containers) or check runs alongside, else 3; stops when STOPCAP exists
W=$(dirname "$0")
while [ ! -e "$W/STOPCAP" ]; do
  n=$(docker ps --format '{{.Names}}' | grep -c 'badminton-match.*sim')
  c=$(docker ps --format '{{.Names}}' | grep -c '^check-badminton')
  echo $((4 - n - c)) > "$W/MAXC"
  sleep 20
done
