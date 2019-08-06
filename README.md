# FLOPS MicroBenchmark

This microbenchmark's goal is to achieve as much FLOP per second as possible, in simple precision (SP), trying to stay close to the theoretical peak performance, on x86 architectures.

To achieve this, there are different executables for each specific CPU microarchitecture, which can be summarized in the following table (you will find them in `output` directory):

##### Intel
| Microarchitecture    | Executable name   |
|:--------------------:|:-----------------:|
| Sandy Bridge         | `sandy_bridge`    |
| Ivy Bridge           | `ivy_bridge`      |
| Haswell              | `haswell`         |
| Broadwell            | `broadwell`       |
| Skylake              | `skylake`         |
| Kaby Lake            | `kaby_lake`       |
| Coffe Lake           | `coffe_lake`      |
| Cannon Lake(AVX2)    | `cannon_lake_256` |
| Cannon Lake(AVX512)  | `cannon_lake_512` |
| Ice Lake(AVX2)       | `ice_lake_256`    |
| Ice Lake(AVX512)     | `ice_lake_512`    |
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
To know it, run this benchmark and compare the results with the theoretical peak performance of your CPU. This can be calculated as:

```
N_CORES*FREQUENCY*2(FMA)*2(Two FMA Units)*(SIZE_OF_VECTOR/32)
```

The size of vector will be 256 if you are using AVX, or 512 if you are using AVX512. For example, for a i7-4790K, we have:

```
4*3.997*10^9*2*2*(256/32) = 511.61 GFLOP/S
```

And, as you can see in the previous test, we got 511.38 GFLOP/S, which tell us test is working good and CPU is behaving exactly as we expected.

## How it works
This test will run vectorized instructions(eg AVX, AVX512) with FMA (if available) in a loop(10^9 times, by default) in parallel, so this will try to achieve the best performance in the current CPU.

## Tests done so far
Here follows a table where I'll be updating results using different processors using this benchmark:

| CPU                       | Peak performance | Results                  | Loss    |
|:-------------------------:|:----------------:|:------------------------:|:-------:|
| Intel i7-4790K (Haswell)  | `511.61 GFLOP/S` | `511.38 +- 0.01 GFLOP/S` | `~0.23` |
| Intel i7-5500U (Broadwell)| `185.53 GFLOP/S` | `185.23 +- 0.05 GFLOP/S` | `~0.30` |
| Intel i5-6400 (Skylake)   | `396.67 GFLOP/S` | `396.61 +- 0.01 GFLOP/S` | `~0.06` |
| AMD Ryzen 2600 (Zen+)     | `357.60 GFLOP/S` | `355.84 +- 0.04 GFLOP/S` | `~1.10` |

## Microarchitecture table

The following table acts as a summary of all supported microarchitectures with their characteristics:


| Microarchitecture    | FMA                | AVX512             | Slots    | FPUs  | Latency |
|:--------------------:|:------------------:|:------------------:|:--------:|:-----:|:-------:|
| Sandy Bridge         | :x:                | :x:                |   5      |     1 |       5 |
| Ivy Bridge           | :x:                | :x:                |   5      |     1 |       5 |
| Haswell              | :heavy_check_mark: | :x:                |  10      |     2 |       5 |
| Broadwell            | :heavy_check_mark: | :x:                |   8      |     2 |       4 |
| Skylake              | :heavy_check_mark: | :x:                |   8      |     2 |       4 |
| Kaby Lake            | :heavy_check_mark: | :x:                | ???      |   ??? |     ??? |
| Coffe Lake           | :heavy_check_mark: | :x:                | ???      |   ??? |     ??? |
| Cannon Lake          | :heavy_check_mark: | :heavy_check_mark: | ???      |   ??? |     ??? |
| Ice Lake             | :heavy_check_mark: | :heavy_check_mark: | ???      |   ??? |     ??? |
| KNL(Knights Landing) | :heavy_check_mark: | :heavy_check_mark: | 12       |     2 |       6 |
| Ryzen ZEN            | :heavy_check_mark: | :x:                | ???      |   ??? |     ??? |
| Ryzen ZEN+           | :heavy_check_mark: | :x:                | ???      |   ??? |     ??? |

This data have been retrieved thanks to [Agner Fog's data](https://www.agner.org/optimize/instruction_tables.pdf) and thanks to [Wikichip](https://en.wikichip.org/wiki/WikiChip). Cells marked with an asterisk (\*) indicates that such data has been obtanied via experimentation because I did not found such information on the web (may be oudated). Cells containing ??? means I don't know this data yet and hence, the results for this microarchitecture may not be optimal (hope this can be improved in the future, when I can have the opportunity to run tests on such microarchitecture).

_NOTE:_ "Slots" column is calculated by means of `FPUs x Latency`.
