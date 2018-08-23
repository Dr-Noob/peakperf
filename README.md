# FLOPS Benchmark

This benchmark's goal is to achieve as much FLOP/s as possible, trying to stay close to the theoretical peak performance. To achieve this, there are different executables for each specific CPU architecture, which can be summarized in the following table:

##### Intel
| Architecture         | Executable file  |
|:--------------------:|:----------------:|
| Haswell(4th Gen)     | `haswell_256`    |
| Broadwell(5th Gen)   | `haswell_256`    |
| Skylake(6th Gen)     | `skylake_256`    |
| KNL(Knights Landing) | `knl_512`        |

##### AMD
| Architecture         | Executable file  |
|:--------------------:|:----------------:|
| Ryzen ZEN            | `zen_plus_256`   |
| Ryzen ZEN+           | `zen_plus_256`   |

__NOTE: This test will not work if you run it under a CPU which hasn't neccesary instructions, like AVX or FMA__

## Usage
Compile and choose the right executable, then run:
__./exec [n_trials n_warmup_trials]__ where both parameters are optional, __n_trials__ is the times the test will be executed and __n_warmup_trials__ the times it will be executed just to warmup(this can be 0 if you prefer no warmup). Then you will see the mean of all executions with a standart derivation, like this:

```
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
This test will run vectorized instructions(eg AVX, AVX512) with FMA in a loop(10^9 times, by default) in parallel, so this will try to achieve the best performance in the current CPU.

## Tests done so far
Here follows a table where I'll be updating results using different processors using this benchmark:

| CPU                       | Peak performance | Results                  | Loss    |
|:-------------------------:|:----------------:|:------------------------:|:-------:|
| Intel i7-4790K(Haswell)   | `511.61 GFLOP/S` | `511.38 +- 0.01 GFLOP/S` | `~0.23` |
| Intel i5-6400(Skylake)    | `396.67 GFLOP/S` | `396.61 +- 0.01 GFLOP/S` | `~0.06` |
