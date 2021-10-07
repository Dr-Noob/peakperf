# peakperf
Microbenchmark to achieve peak performance on x86_64 CPUs and NVIDIA GPUs.

**Table of Contents**
<!-- UPDATE with: doctoc --notitle README.md -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [1. Support](#1-support)
  - [1.1 Software support](#11-software-support)
  - [1.2 Hardware support](#12-hardware-support)
- [2. Instalation](#2-instalation)
  - [2.1 Building from source](#21-building-from-source)
  - [2.2 Enabling and disabling support for CPU/GPU](#22-enabling-and-disabling-support-for-cpugpu)
    - [CUDA is installed but peakperf is unable to find it](#cuda-is-installed-but-peakperf-is-unable-to-find-it)
    - [Manually disabling compilation for CPU/GPU](#manually-disabling-compilation-for-cpugpu)
- [3. Usage:](#3-usage)
  - [3.1 Selecting CPU or GPU](#31-selecting-cpu-or-gpu)
  - [3.2. The environment](#32-the-environment)
  - [3.3. Microarchitecture detection](#33-microarchitecture-detection)
  - [3.4. Options](#34-options)
- [4. Understanding the microbenchmark](#4-understanding-the-microbenchmark)
  - [4.1 What is "peak performance" anyway?](#41-what-is-peak-performance-anyway)
  - [4.2 The formula](#42-the-formula)
  - [4.3 About the frequency to use in the formula](#43-about-the-frequency-to-use-in-the-formula)
  - [4.4 What can I do if I do not get the expected results?](#44-what-can-i-do-if-i-do-not-get-the-expected-results)
- [5. Evaluation](#5-evaluation)
  - [Intel](#intel)
  - [AMD](#amd)
  - [NVIDIA](#nvidia)
- [6. Microarchitecture table](#6-microarchitecture-table)
  - [6.1 CPU](#61-cpu)
  - [6.2 GPU](#62-gpu)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. Support

## 1.1 Software support
peakperf only works properly in Linux. peakperf under Windows / macOS has not been tested, so performance may not be optimal. Windows port may be implemented in the future (see [Issue #1](https://github.com/Dr-Noob/peakperf/issues/1))

## 1.2 Hardware support
Supported microarchitectures are:

- **CPU (x86_64)**:
  - Intel: Sandy Bridge	and newer.
  - AMD: Zen and newer.
- **GPU**:
  - NVIDIA: Compute Capabitliy >= 2.0.

For a complete list of supported microarchitectures, see section [5](#5-evaluation).

NOTES:
- _Only GPUs that support to read the freqeuncy in real time (using freq.sh) can be actually evaluated._
- _Other microarchitectures not mentioned here may also work._

# 2. Instalation
There is a peakperf package available in Arch Linux ([peakperf-git](https://aur.archlinux.org/packages/peakperf-git)).

If you are in another distro, you can build `peakperf` from source.

## 2.1 Building from source
Build the microbenchmark with the build script, which uses `cmake`:

```
git clone https://github.com/Dr-Noob/peakperf
cd peakperf
./build.sh
./peakperf
```

## 2.2 Enabling and disabling support for CPU/GPU
By default, peakperf will be built with support for CPU and GPU. The support for the GPU will only be enabled if CUDA is found. During the `cmake` execution, peakperf will print a summary where you can check which devices peakperf was compiled for.

```
-- ----------------------
-- peakperf build report:
-- CPU mode: ON
-- GPU mode: ON
-- ----------------------
```

### CUDA is installed but peakperf is unable to find it
Sometimes, `cmake` will fail to find CUDA even tough it is installed. To let `cmake` find CUDA, edit the build.sh script and use:
- `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`
- `-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/path/to/cuda`

### Manually disabling compilation for CPU/GPU
Use `cmake` variables:
- `-DENABLE_CPU_DEVICE=[ON|OFF]`
- `-DENABLE_GPU_DEVICE=[ON|OFF]`

For example, building with `-DENABLE_CPU_DEVICE=OFF` results in:

```
-- ----------------------
-- peakperf build report:
-- CPU mode: OFF
-- GPU mode: ON
-- ----------------------
```

# 3. Usage:

## 3.1 Selecting CPU or GPU
By default, peakperf will run on the CPU:

```
[noob@drnoob peakperf]$ ./peakperf -t 4

-----------------------------------------------------
    peakperf (https://github.com/Dr-Noob/peakperf)
-----------------------------------------------------
        CPU: Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz
  Microarch: Haswell
  Benchmark: Haswell (AVX2)
 Iterations: 1.00e+09
      GFLOP: 640.00
    Threads: 4

   Nº  Time(s)  GFLOP/s
    1  1.25743   508.97 *
    2  1.25137   511.44 *
    3  1.25141   511.42
    ...................
   12  1.25136   511.44
-----------------------------------------------------
 Average performance:      511.43 +- 0.01 GFLOP/s
-----------------------------------------------------
* - warm-up, not included in average
```

To manually select the device, use `-d [cpu|gpu]`. To run peakpef on the GPU:

```
[noob@drnoob peakperf]$ ./peakperf -d gpu

------------------------------------------------------
    peakperf (https://github.com/Dr-Noob/peakperf)
------------------------------------------------------
           GPU: GeForce GTX 970
     Microarch: Maxwell
    Iterations: 4.00e+08
         GFLOP: 7987.20
        Blocks: 13
 Threads/block: 768

   Nº  Time(s)  GFLOP/s
    1  1.87078  4269.44 *
    2  1.84159  4337.12 *
    3  1.84205  4336.03
    ...................
   12  1.84194  4336.31
------------------------------------------------------
 Average performance:      4336.69 +- 0.91 GFLOP/s
------------------------------------------------------
* - warm-up, not included in average
```

## 3.2. The environment
To achieve the best performance, you should run this test with the computer working under minimum load (e.g, in non-graphics mode). If you are in a desktop machine, a good way to do this is by issuing `systemctl isolate multi-user.target`.

## 3.3. Microarchitecture detection
peakperf automatically detects your CPU/GPU and runs the best benchmark for your architecture.

1. For the CPU, you can see all available benchmarks in peakperf and select which one you one to run:

```
[noob@drnoob peakperf]$ ./peakperf -l
Available benchmark types:
...
[noob@drnoob peakperf]$ ./peakperf -b haswell
```

2. For the GPU, only one benchmark exists, and the optimality of the microbenchmark depends on the kernel launch configuration. peakperf automatically determine this configuration for your GPU.

## 3.4. Options
peakperf has many different options to tweak and expriment with your hardware. Use `-h` to print all available options

_NOTE_: Some options are available only on CPU or GPU

# 4. Understanding the microbenchmark
## 4.1 What is "peak performance" anyway?
Peak performance refers to the maximum performance that a chip (a CPU) can achieve. The more powerful the CPU is, the greater the peak performance can achieve. This performance is a theoretical limit, computed using a formula (see next section), measured in floating point operation per seconds (FLOP/s or GFLOP/s, which stands for gigaflops). This value establishes a performance limit that the CPU is unable to overcome. However, achieving the peak performance (the maximum performance for a given CPU) is a very hard (but also interesting) task. To do so, the software must take advantage of the full power of the CPU. peakperf is a microbenchmark that achieves peak performance on many different x86_64 microarchitectures.

## 4.2 The formula

```
N_CORES * FREQUENCY * FMA * UNITS * (SIZE_OF_VECTOR/32)
```

- N_CORES: The number of physical cores. In our example, it is **4**
- FREQUENCY: The freqeuncy of the CPU measured in GHz. To measure this frequency is a bit tricky, see next section for more details. In our example, it is **3.997**.
- FMA: If CPU supports FMA, the peak performance is multipled by 2. If not, it is multiplied by 1. In our example, it is **2**.
- UNITS: CPUs can provide 1 or 2 functional units per core. Modern Intel CPUs usually provide 2, while AMD CPUs usually provide 1. In our example, it is **2**.
- SIZE_OF_VECTOR: If CPU supports AVX, the size is 256 (because AVX is 256 bits long). If CPU supports AVX512, the size is 512. In our example, the size is **256**.

For the example of a i7-4790K, we have:

```
4 * 3.997 * 10^9 * 2 * 2 * (256/32) = 511.61 GFLOP/s
```

And, as you can see in the previous test, we got 511.43 GFLOP/S, which tell us that peakperf is working properly and our CPU is behaving exactly as we expected. But, why did I chosse 3.997 GHz as the frequency?

## 4.3 About the frequency to use in the formula

While running this microbenchmark, your CPU will be executing AVX code, so the frequency of your CPU running this code is neither your base nor your turbo frequency. Please, have a look at [this document](http://www.dolbeau.name/dolbeau/publications/peak.pdf) (on section IV.B) for more information.

The AVX frequency for a specific CPU is sometimes available online. The most effective way I know to get this frequency is to to actually measure your CPU frequency on real time while running AVX code. You can use the script I crafted for this task, [freq.sh](https://github.com/Dr-Noob/peakperf/freq.sh), to achieve this:
1. Run the microbenchmark in background (`./peakperf -r 4 -w 0 > /dev/null &`)
2. Run the script (`./freq.sh`) which will fetch your CPU frequency in real time. In my case, I get:

```
Every 0,2s: grep 'MHz' /proc/cpuinfo

cpu MHz         : 3997.629
cpu MHz         : 3997.629
cpu MHz         : 3997.630
cpu MHz         : 3997.630
cpu MHz         : 3997.630
cpu MHz         : 3997.630
cpu MHz         : 3997.629
cpu MHz         : 3997.630
```

As you can see, i7-4790K's frequency while running AVX code is ~3997.630 MHz, which equals to 3.997 GHz. However, you may see that your frequency fluctuates too much, so that it's impossible to estimate the frequency of your CPU. This may happen because:
1. The microbenchmark is not working correctly. Please create a [issue in github](https://github.com/Dr-Noob/peakperf/issues)
2. Your CPU is not able to keep a stable frequency. This often happens if it's to hot, so the CPU is forced to low the frequency to not to melt itself.

## 4.4 What can I do if I do not get the expected results?
Please create a [issue in github](https://github.com/Dr-Noob/peakperf/issues), posting the output of peakperf.

# 5. Evaluation
This tables shows the performance of peakperf for each of the microarchitecture supported by the microbenchmark. **To see all the hardware tested, see [benchmarks](BENCHMARKS.md)**

## Intel
| uarch           | CPU                | AVX Clock    | PP (Formula) | PP (Experimental)  | Loss    |
|:---------------:|:------------------:|:------------:|:------------:|:------------------:|:-------:|
| Sandy Bridge    | i5-2400            | `3.192 GHz`  |  `102.14`    |  `100.64 +- 0.00`  | `1.46%` |
| Ivy Bridge      | 2x Xeon E5-2650 v2 | `2.999 GHz`  |  `767.74`    |  `744.24 +- 3.85`  | `3.15%` |
| Haswell         | i7-4790K           | `3.997 GHz`  |  `511.61`    |  `511.43 +- 0.01`  | `0.03%` |
| Broadwell       | 2x Xeon E5-2698 v4 | `2.599 GHz`  | `3326.72`    | `3269.87 +- 14.42` | `1.73%` |
| Skylake         | i5-6400            | `3.099 GHz`  |  `396.67`    |  `396.61 +- 0.01 ` | `0.06`  |
| Knights Landing | Xeon Phi 7250      | `1.499 GHz`  | `5991.69`    | `5390.84 +- 7.83`  | `3.72%` |
| Kaby Lake       | i5-8250U           | `2.700 GHz`  |  `345.60`    |  `343.57 +- 1.38`  | `0.59%` |
| Coffee Lake     | i9-9900K           | `3.600 GHz`  |  `921.60`    |  `918.72 +- 1.13`  | `0.31%` |
| Comet Lake      | i9-10900KF         | `4.100 GHz`  |  `1312.00`   | `1308.24 +- 0.30`  | `0.30%` |
| Cascade Lake    | 2x Xeon Gold 6238  | `2.099 GHz`  |  `5910.78`   | `5851.60 +- 2.69`  | `1.01%` |
| Ice Lake        | i5-1035G1          | `2.990 GHz`  |  `382.72`    |  `382.22 +- 0.18`  | `0.13%` |
| Tiger Lake      | -                  | -            |  -           |  -                 | -       |
| Rocket Lake     | -                  | -            |  -           |  -                 | -       |

## AMD
| uarch | CPU              | AVX Clock    | PP (Formula) | PP (Experimental)  | Loss    |
|:-----:|:----------------:|:------------:|:------------:|:------------------:|:-------:|
| Zen   | -                | -            | -            | -                  | -       |
| Zen+  | AMD Ryzen 5 2600 | `3.724 GHz`  | `357.50`     | `357.08 +- 0.03`   | `0.11%` |
| Zen 2 | -                | -            | -            | -                  | -       |
| Zen 3 | -                | -            | -            | -                  | -       |

## NVIDIA
| C.C | uarch   | GPU         | Clock        | PP (Formula) | PP (Experimental)   | Loss    |
|:---:|:-------:|:-----------:|:------------:|:------------:|:-------------------:|:-------:|
| 5.2 | Maxwell | GTX 970     | `1.341 GHz`  | `4462.84`    | `4333.92 +- 0.90`   | `2.97%` |
| 6.1 | Pascal  | GTX 1080    | `1.860 GHz`  | `9523.20`    | `9397.97 +- 0.10`   | `1.33%` |
| 7.5 | Turing  | RTX 2080 Ti | `1.905 GHz`  | `16581.12`   | `16373.28 +- 16.07` | `1.26%` |
| 8.6 | Ampere  | -           | -            | -            | -                   | -       |

_NOTE 1_: Performance measured on simple precision and GFLOP/s (gigaflops per second).

_NOTE 2_: The clock information is retrieved _experimentally_. In other words, this data is not the theoretical values for each device, but the actual frequency measured on each device (using `freq.sh` script).

_NOTE 3_: KNL performance is computed as PP * (6/7) (see [explanation](https://sites.utexas.edu/jdm4372/2018/01/22/a-peculiar-throughput-limitation-on-intels-xeon-phi-x200-knights-landing/)).

_NOTE 4_: Sandy Bridge and Ivy Bridge have ADD and MUL VPUs that can be used in parallel. Therefore, Xeon E5-2650 v2 formula is computed as `FREQ * CORES * 2 * 2 * 8`. However, i5-2400 peak performance is computed as the half. The explanation for this is that ADD and MUL VPUs can only be used if CPU supports hyperthreading. If CPU do not support hyperthreading, one core is unable to fill both VPUs fast enough.

# 6. Microarchitecture table

The following tables act as a summary of all supported microarchitectures with their characteristics.

## 6.1 CPU
| uarch           | FMA              | AVX512             | Slots | FPUs            | Latency         | Tested           | Refs |
|:---------------:|:----------------:|:------------------:|:-----:|:---------------:|:---------------:|:----------------:|:----:|
| Sandy Bridge    | :x:              | :x:                |     6 | 2 (ADD+MUL AVX) | 3 (ADD) 5 (MUL) |:heavy_check_mark:|  [1] |
| Ivy Bridge      | :x:              | :x:                |     6 | 2 (ADD+MUL AVX) | 3 (ADD) 5 (MUL) |:heavy_check_mark:|  [2] |
| Haswell         |:heavy_check_mark:| :x:                |    10 | 2 (FMA AVX2)    | 5 (FMA)         |:heavy_check_mark:|  [3] |
| Broadwell       |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:|  [3] |
| Skylake         |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:|  [3] |
| Kaby Lake       |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:|  [4] |
| Coffee Lake     |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:|  [5] |
| Comet Lake      |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:| [10] |
| Ice Lake        |:heavy_check_mark:| :heavy_check_mark: |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:| [12] |
| Tiger Lake      |:heavy_check_mark:| :heavy_check_mark: |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:heavy_check_mark:| [12] |
| Knights Landing |:heavy_check_mark:| :heavy_check_mark: |    12 | 2 (FMA AVX512)  | 6 (FMA)         |:heavy_check_mark:|  [6] |
| Piledriver      |:heavy_check_mark:| :x:                |     5 | 1 (FMA AVX)     | 5 (FMA)         |:x:               |  [?] |
| Zen             |:heavy_check_mark:| :x:                |     5 | 1 (FMA AVX2)    | 5 (FMA)         |:x:               |  [7] |
| Zen+            |:heavy_check_mark:| :x:                |     5 | 1 (FMA AVX2)    | 5 (FMA)         |:heavy_check_mark:|  [8] |
| Zen 2           |:heavy_check_mark:| :x:                |    10 | 2 (FMA AVX2)    | 5 (FMA)         |:x:               |  [9] |
| Zen 3           |:heavy_check_mark:| :x:                |     8 | 2 (FMA AVX2)    | 4 (FMA)         |:x:               | [11] |

References:
- [1]  [Agner Fog Instruction Tables (Page 199, VADDPS)](https://www.agner.org/optimize/instruction_tables.pdf)
- [2]  [Agner Fog Instruction Tables (Page 213, VADDPS)](https://www.agner.org/optimize/instruction_tables.pdf)
- [3]  [Intel Intrinsics Guide (_mm256_fmadd_ps)](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=256_fmadd_ps&expand=136,2553)
- [4]  [Wikichip](https://en.wikichip.org/wiki/intel/microarchitectures/kaby_lake#Pipeline)
- [5]  [Agner Fog Instruction Tables (Page 299, VFMADD)](https://www.agner.org/optimize/instruction_tables.pdf)
- [6]  [Intel Intrinsics Guide (_mm512_fmadd_ps)](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_fmadd_ps&expand=136,2553,2557)
- [7]  [Agner Fog Instruction Tables (Page 99, VFMADD)](https://www.agner.org/optimize/instruction_tables.pdf)
- [8]  [Wikichip](https://en.wikichip.org/wiki/amd/microarchitectures/zen%2B#Pipeline)
- [9]  [Agner Fog Instruction Tables (Page 111, VFMADD132PS)](https://www.agner.org/optimize/instruction_tables.pdf)
- [10] [Wikichip](https://en.wikichip.org/wiki/intel/microarchitectures/comet_lake)
- [11]  [Agner Fog Instruction Tables (Page 124, VFMADD132PS)](https://www.agner.org/optimize/instruction_tables.pdf)
- [12]  [Agner Fog Instruction Tables (Page 347, VADDPS)](https://www.agner.org/optimize/instruction_tables.pdf)

## 6.2 GPU
| uarch   | Latency  | Tested           | Refs |
|:-------:|:--------:|:----------------:|:----:|
| Maxwell |  6       |:heavy_check_mark:|  [] |
| Pascal  |  6       |:heavy_check_mark:|  [] |
| Turing  |  4       |:heavy_check_mark:|  [] |
| Ampere  |  ?       |:x:               |  [] |

_NOTES:_
- The fact that a CPU belongs to a microarchitecture does not imply that it supports the vector extensions shown in this table (e.g, Pentium Skylake does not support AVX).
- Older microarchitectures may be added in the future. If I have not added olds architecture is because I can't test peakperf on them since I have not access to this hardware.
- Ice Lake and Tiger Lake support AVX512 instructions but they only have 1 AVX512 VPU (at least in client versions), while it has 2 VPUs for AVX2. Because AVX512 runs in lower freqeuncy, the performance obtained with AVX2 (using 2 VPUs) is better than with AVX512 (using 1 VPU). Thus, peak performance is obtained using AVX2, although it supports AVX512 instruction set.
- Slots column is calculated with `FPUs x Latency`.
