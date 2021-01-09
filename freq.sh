#!/bin/bash

watch -n0.2 "grep 'MHz' /proc/cpuinfo | sort -nr && sensors"
