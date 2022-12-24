#!/bin/ksh
for (( i=60; i>0; i--)); do
  sleep 1 &
  printf "  $i \r"
  wait
done
