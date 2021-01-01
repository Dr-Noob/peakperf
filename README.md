# peakperf

This microbenchmark's goal is to achieve as much FLOP per second as possible, in simple precision (SP), trying to stay close to the theoretical peak performance, on x86 architectures.

To achieve this, there are different executables for each specific CPU microarchitecture, which can be summarized in the following table (you will find them in `output` directory):

##### Intel
| Microarchitecture    | Executable name   |
|:--------------------:|:-----------------:|
| Sandy Bridge         | `sandy_bridge`    |
| Ivy Bridge           | `ivy_bridge`      |
| Haswell              | `haswell`         |
| Broadwell            | `broadwell`       |
| Skylake              | `skylake_256`     |
| Cascade Lake(AVX512) | `skylake_512`     |
| Kaby Lake            | `kaby_lake`       |
| Coffe Lake           | `coffe_lake`      |
| Cannon Lake (AVX2)   | `cannon_lake_256` |
| Cannon Lake (AVX512) | `cannon_lake_512` |
| Ice Lake (Client)    | `ice_lake_256`    |
| KNL(Knights Landing) | `knl`             |

##### AMD
| Architecture         | Executable file  |
|:--------------------:|:----------------:|
| Ryzen ZEN            | `zen`            |
| Ryzen ZEN+           | `zen_plus`       |

__NOTE: This test will not work if you run it under a CPU which hasn't neccesary instructions, like AVX or FMA__

## Usage
Compile with `make` and choose the right executable, according to the previous table. If your CPU's microarchitecture is not included in the table, please write an issue and I will consider adding it to the project.

The microbenchmark has several options, which can be checked via `-h`:

```
[noob@drnoob FLOPS]$ ./haswell -h
Usage: ./haswell [-h] [-r n_trials] [-w warmup_trials] [-t n_threads]
    Options:
      -h      Print this help and exit
      -r      Set the number of trials of the benchmark
      -w      Set the number of warmup trials
      -t      Set the number of threads to use
```

All of them are optional, so you can run the microbenchmark without arguments:

```
[noob@drnoob FLOPS]$ ./haswell
Benchmarking FLOPS by Dr-Noob(github.com/Dr-Noob/FLOPS).
   Test name: Haswell - 256 bits
         CPU: Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz
  Iterations: 1000000000
       GFLOP: 1280.00
     Threads: 8

   Nº  Time(s)  GFLOP/S
    1  2.51006   509.95 *
    2  2.50301   511.38
    3  2.50302   511.38
    4  2.50321   511.34
    5  2.50301   511.38
    6  2.50300   511.39
    7  2.50301   511.38
-------------------------------------------------
Average performance:      511.38 +- 0.01 GFLOP/s
-------------------------------------------------
* - warm-up, not included in average
```

To achieve the best results, you should run this test with the computer working under minimum load. If you're using Linux, a good way to do this is by issuing `systemctl isolate multi-user.target`.

## Is my CPU behaving as it should?
#### 1. The formula

To know if you are reaching the peak performance of your CPU, run this benchmark and compare the experimental results with the theoretical peak performance of your CPU. This can be calculated as:

```
N_CORES*FREQUENCY*2(FMA)*2(Two FMA Units)*(SIZE_OF_VECTOR/32)
```

The size of vector will be 256 if you are using AVX, or 512 if you are using AVX512. You have to take into account if you have 1 or 2 units, or if you have FMA or not. For example, for a i7-4790K, we have:

```
4*3.997*10^9*2*2*(256/32) = 511.61 GFLOP/S
```

And, as you can see in the previous test, we got 511.38 GFLOP/S, which tell us test is working good and CPU is behaving exactly as we expected. But, why did I chosse 3.997 GHz as the frequency?

#### 2. About the frequency to use in the formula

While running this microbenchmark, your CPU will be executing AVX code, so the frequency of your CPU running this code is neither your base nor your turbo frequency. Please, have a look at [this document](http://www.dolbeau.name/dolbeau/publications/peak.pdf) (on section IV.B) for more information.

The vendor of your processor may have published this frequency but this happens rarely. The most effective way I know to get this frequency is to to actually measure your CPU frequency on real time while running AVX code. You can use the script [freq.sh](https://github.com/Dr-Noob/FLOPS/freq.sh) to achieve this:
1. Run the microbenchmark in background (`output/microbench -r 4 -w 0 > /dev/null &`)
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

As you can see, i7-4790K's frequency while running AVX code is ~3997.630 MHz, which equals to 3.997 GHz. However, you may see that your frequency fluctuates too much, so that it's impossible to estimate the frequency of your CPU. This may happen due to reasons:
1. The microbenchmark is not working correctly. You may contact me to try to fix the problem.
2. Your CPU is not able to keep a stable frequency. This often happens if it's to hot, so the CPU is forced to low the frequency to not to melt itself.

#### 3. What if I do not get the expected results?
You should contact me to fix the problem!

## How this microbenchmark works
This test will run vectorized instructions(eg AVX, AVX512) with FMA (if available) in a loop(10^9 times, by default) in parallel, so this will try to achieve the best performance in the current CPU.

## Tests done so far
Here follows a table where I'll be updating results using different processors using this benchmark.

| CPU (Desktop CPUs)             | AVX Freq     | PP (Formula)      | PP (Experimental)        | Loss    |
|:------------------------------:|:------------:|:-----------------:|:------------------------:|:-------:|
| Intel i5-2400  (Sandy Bridge ) | `3.192 GHz`  | `102.14 GFLOP/S`  | `100.64 +- 0.00 GFLOP/S` | `1.46%` |
| Intel i7-4790K (Haswell)       | `3.997 GHz`  | `511.61 GFLOP/S`  | `511.43 +- 0.01 GFLOP/S` | `0.03%` |
| Intel i7-5500U (Broadwell)     | `2.895 GHz`  | `185.28 GFLOP/S`  | `183.03 +- 0.28 GFLOP/S` | `1.21%` |
| Intel i5-8250U (Kaby Lake)     | `2.700 GHz`  | `345.60 GFLOP/S`  | `343.57 +- 1.38 GFLOP/S` | `0.59%` |
| Intel i7-8700 (Coffe Lake)     | `4.300 GHz`  | `825.60 GFLOP/S`  | `823.83 +- 0.01 GFLOP/S` | `0.21%` |
| Intel i9-9900K (Coffe Lake)    | `3.600 GHz`  | `921.60 GFLOP/S`  | `918.72 +- 1.13 GFLOP/S` | `0.31%` |
| Intel i5-1035G1 (Ice Lake)     | `2.990 GHz`  | `382.72 GFLOP/S`  | `382.22 +- 0.18 GFLOP/S` | `0.13%` |
| AMD Ryzen 5 2600 (Zen+)        | `3.724 GHz`  | `357.50 GFLOP/S`  | `357.08 +- 0.03 GFLOP/S` | `0.11%` |

| CPU (HPC / Server CPUs)            | AVX Freq     | PP (Formula)      | PP (Experimental)          | Loss     |
|:----------------------------------:|:------------:|:-----------------:|:--------------------------:|:--------:|
| Intel Xeon Phi KNL 7250 (KNL)      | `1.499 GHz`  | `5991.69 GFLOP/S` | `5390.84 +- 7.83 GFLOP/s`  | `3.72%` |
| 2x Xeon E5-2650 v2 (Ivy Bridge)    | `2.999 GHz`  | `383.87 GFLOP/S`  | `377.66 +- 0.02 GFLOP/s`   | `1.64%`  |
| 2x Xeon E5-2698 v4 (Broadwell)     | `2.599 GHz`  | `3326.72 GFLOP/S` | `3269.87 +- 14.42 GFLOP/s` | `1.73%`  |
| 2x Xeon Gold 6238 (Cascade Lake)   | `2.099 GHz`  | `5910.78 GFLOP/S` | `5851.60 +- 2.69 GFLOP/s`  | `1.01%`  |
| 2x Xeon Gold 6226R (Cascade Lake)  | `2.499 GHz`  | `5117.95 GFLOP/S` | `5088.06 +- 0.33 GFLOP/s`  | `0.58%`  |

_NOTE 1_: On some machines, I'm not root or even the person running the microbenchmark, in which case, the possible overhead (because of not running the microbenchmark in a adequate environment) may deteriorate the results.

_NOTE 2_: KNL performance is computed as PP * (6/7) (see [explanation](https://sites.utexas.edu/jdm4372/2018/01/22/a-peculiar-throughput-limitation-on-intels-xeon-phi-x200-knights-landing/)).

## Microarchitecture table

The following table acts as a summary of all supported microarchitectures with their characteristics:

| Microarchitecture     | FMA                | AVX512             | Slots    | FPUs               | Latency      | Tested            |
|:---------------------:|:------------------:|:------------------:|:--------:|:------------------:|:------------:|:-----------------:|
| Sandy Bridge          | :x:                | :x:                |   3      |     1 (ADD AVX2)   |       3 (ADD)|:heavy_check_mark: |
| Ivy Bridge            | :x:                | :x:                |   3      |     1 (ADD AVX2)   |       3 (ADD)|:heavy_check_mark: |
| Haswell               | :heavy_check_mark: | :x:                |  10      |     2 (FMA AVX2)   |       5 (FMA)|:heavy_check_mark: |
| Broadwell             | :heavy_check_mark: | :x:                |   8      |     2 (FMA AVX2)   |       4 (FMA)|:heavy_check_mark: |
| Skylake               | :heavy_check_mark: | :x:                |   8      |     2 (FMA AVX2)   |       4 (FMA)|:heavy_check_mark: |
| Kaby Lake             | :heavy_check_mark: | :x:                |  8*      |    2* (FMA AVX2)   |      4* (FMA)|:heavy_check_mark: |
| Coffe Lake            | :heavy_check_mark: | :x:                | 10*      |    2* (FMA AVX2)   |      5* (FMA)|:heavy_check_mark: |
| Cannon Lake           | :heavy_check_mark: | :heavy_check_mark: | ???      |   ??? (FMA AVX512) |     ??? (FMA)|:x:                |
| Ice Lake              | :heavy_check_mark: | :heavy_check_mark: | 8        |     2 (FMA AVX2)   |       4 (FMA)|:heavy_check_mark: |
| KNL (Knights Landing) | :heavy_check_mark: | :heavy_check_mark: | 12       |     2 (FMA AVX512) |       6 (FMA)|:heavy_check_mark: |
| Ryzen ZEN             | :heavy_check_mark: | :x:                | 5        |     1 (FMA AVX2)   |       5 (FMA)|:x:                |
| Ryzen ZEN+            | :heavy_check_mark: | :x:                | 5        |     1 (FMA AVX2)   |       5 (FMA)|:heavy_check_mark: |

This data have been retrieved thanks to [Agner Fog's data](https://www.agner.org/optimize/instruction_tables.pdf),[Wikichip](https://en.wikichip.org/wiki/WikiChip) and [Intel's Intrisics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide).
- Cells marked with an asterisk (\*) indicates that such data has been obtanied via experimentation because I did not found such information on the web (may be oudated).
- Cells containing ??? means I don't know this data yet and hence, the results for this microarchitecture may not be optimal (hope this can be improved in the future, when I can have the opportunity to run tests on such microarchitecture).
- **Ice Lake:** While it has AVX512 instructions, it only has 1 AVX512 VPU (at least in client Ice Lake), while it has 2 VPUs for AVX2. Because AVX512 runs in lower freqeuncy, the performance obtained with AVX2 (using 2 VPUs) is better than with AVX512 (using 1 VPU). Thus, peak performance in Ice Lake is obtained using AVX2, although it supports AVX512 instruction set.

_NOTE:_ "Slots" column is calculated by means of `FPUs x Latency`.
