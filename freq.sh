#!/bin/bash

if [ "$1" == "gpu" ]
then
  # nvidia-smi --help-query-gpu
  # sudo nvidia-smi -i 0 -pm 1
  watch -n0.2 "nvidia-smi -i 0 --query-gpu=utilization.gpu,clocks.current.sm,pstate,fan.speed,temperature.gpu --format=csv"
else
  watch -n0.2 "grep 'MHz' /proc/cpuinfo | sort -nr && sensors"
fi
