#!/bin/bash

if [ "$1" == "gpu" ]
then
  watch -n0.5 "nvidia-smi -i 0 --query-gpu=clocks.current.sm,pstate,fan.speed,temperature.gpu --format=csv"
else
  watch -n0.2 "grep 'MHz' /proc/cpuinfo | sort -nr && sensors"
fi
