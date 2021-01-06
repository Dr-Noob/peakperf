## Tests on real hardware (Desktop CPUs)

| CPU              | uArch        | AVX Freq     | PP (Formula) | PP (Experimental) | Loss    |
|:----------------:|:------------:|:------------:|:------------:|:-----------------:|:-------:|
| Intel i5-2400    | Sandy Bridge | `3.192 GHz`  | `102.14`     | `100.64 +- 0.00`  | `1.46%` |
| Intel i7-4790K   | Haswell      | `3.997 GHz`  | `511.61`     | `511.43 +- 0.01`  | `0.03%` |
| Intel i7-5500U   | Broadwell    | `2.895 GHz`  | `185.28`     | `183.03 +- 0.28`  | `1.21%` |
| Intel i5-6400    | Skylake      | `3.099 GHz`  | `396.67`     | `396.61 +- 0.01 ` | `0.06`  |
| Intel i5-8250U   | Kaby Lake    | `2.700 GHz`  | `345.60`     | `343.57 +- 1.38`  | `0.59%` |
| Intel i7-8700    | Coffee Lake  | `4.300 GHz`  | `825.60`     | `823.83 +- 0.01`  | `0.21%` |
| Intel i9-9900K   | Coffee Lake  | `3.600 GHz`  | `921.60`     | `918.72 +- 1.13`  | `0.31%` |
| Intel i5-1035G1  | Ice Lake     | `2.990 GHz`  | `382.72`     | `382.22 +- 0.18`  | `0.13%` |
| AMD Ryzen 5 2600 | Zen+         | `3.724 GHz`  | `357.50`     | `357.08 +- 0.03`  | `0.11%` |

## Tests on real hardware (HPC / Server CPUs)
| CPU                     | uArch        | AVX Freq     | PP (Formula) | PP (Experimental)  | Loss     |
|:-----------------------:|:------------:|:------------:|:------------:|:------------------:|:--------:|
| Intel Xeon Phi KNL 7250 | KNL          | `1.499 GHz`  | `5991.69`    | `5390.84 +- 7.83`  | `3.72%`  |
| 2x Xeon E5-2650 v2      | Ivy Bridge   | `2.999 GHz`  | `383.87`     | `377.66 +- 0.02`   | `1.64%`  |
| 2x Xeon E5-2698 v4      | Broadwell    | `2.599 GHz`  | `3326.72`    | `3269.87 +- 14.42` | `1.73%`  |
| 2x Xeon Gold 6238       | Cascade Lake | `2.099 GHz`  | `5910.78`    | `5851.60 +- 2.69`  | `1.01%`  |
| 2x Xeon Gold 6226R      | Cascade Lake | `2.499 GHz`  | `5117.95`    | `5088.06 +- 0.33`  | `0.58%`  |