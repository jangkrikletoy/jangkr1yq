#!/bin/ksh
for (( i=60; i>0; i--)); do
  sleep 1 &
  echo "  $i \r"
  wait
done
