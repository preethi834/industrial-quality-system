# Cost-aware decisions (step 6)

Euro values are **assumptions**, not measurements; results are on public benchmark
images, not plant data. Every euro figure comes from `configs/costs/example_costs.yaml` sha256 `e860bf58b48c1350492238bee5d4d25424500ffa7c6da6b8bf46dbe366b8114d`.

Assumed: false reject €5, escape €200, review €0.5, reviewer miss 5%, reviewer false reject 2%, production defect rate 1%. Cut-offs on p: accept ≤ 0.0032, reject ≥ 0.2953 (unconstrained); above q̂ reject iff p ≥ 0.2953 (guaranteed).

Policies: **guaranteed** = accept iff score ≤ q̂ (step-2 conformal threshold), else
reject or review by expected cost; **unconstrained** = cost-optimal on p, *no*
false-alarm guarantee; **fixed_conformal** = reject iff score > q̂, no review;
**fixed_provisional** = step-1 threshold (largest calibration score), no review.
Intervals: 2000 stratified bootstrap resamples of sup_test
(calibrator and q̂ fixed: test-sample noise only).

## Pooled (equal weight over 17 categories) (costs sha256:e860bf58b48c)

| policy | € per 10,000 parts [95 % CI] | review share | flag share |
|---|---|---|---|
| guaranteed | 3,405 [2,889, 3,957] (costs sha256:e860bf58b48c) | 10.79% | 11.27% |
| unconstrained | 1,903 [1,528, 2,313] (costs sha256:e860bf58b48c) | 14.62% | 15.10% |
| fixed_conformal | 7,547 [5,536, 9,752] (costs sha256:e860bf58b48c) | 0.00% | 11.27% |
| fixed_provisional | 7,619 [5,609, 9,840] (costs sha256:e860bf58b48c) | 0.00% | 11.26% |

Saving of guaranteed vs fixed_conformal: €4,143 [2,361, 6,057] per 10,000 parts (costs sha256:e860bf58b48c).

Saving of unconstrained vs fixed_conformal: €5,645 [3,718, 7,742] per 10,000 parts (costs sha256:e860bf58b48c).

Realised false-alarm rate of the guaranteed policy (all official good test images, categories where the guarantee is claimed): 14/495 = 0.028 [0.016, 0.047], mean α 0.035.

## Per category (costs sha256:e860bf58b48c)

| category | α | guarantee | guaranteed € [CI] | unconstrained € [CI] | fixed q̂ € [CI] | step-1 € | review share | flag rate (all good test) |
|---|---|---|---|---|---|---|---|---|
| mvtec_ad/bottle | 0.05 | claimed | 553 [332, 774] | 553 [332, 774] | 0 [0, 0] | 0 | 0.53% | 0/20 = 0.000 |
| mvtec_ad/cable | 0.05 | claimed | 1,766 [300, 3,910] | 1,111 [338, 2,197] | 1,429 [0, 3,571] | 1,429 | 0.32% | 0/58 = 0.000 |
| mvtec_ad/capsule | 0.05 | claimed | 1,880 [636, 3,667] | 1,880 [636, 3,667] | 1,212 [0, 3,030] | 2,424 | 0.64% | 3/23 = 0.130 |
| mvtec_ad/carpet | 0.02 | not claimed (#17) | 4,177 [1,911, 6,520] | 1,909 [389, 4,054] | 25,491 [6,928, 43,331] | 25,491 | 49.94% | 9/28 = 0.321 |
| mvtec_ad/grid | 0.02 | claimed | 2,784 [556, 6,004] | 2,659 [618, 4,825] | 9,426 [0, 25,926] | 9,426 | 17.09% | 1/21 = 0.048 |
| mvtec_ad/hazelnut | 0.02 | claimed | 578 [368, 788] | 578 [368, 788] | 0 [0, 0] | 0 | 0.55% | 0/40 = 0.000 |
| mvtec_ad/leather | 0.02 | not claimed (#17) | 3,864 [2,044, 5,646] | 300 [150, 488] | 29,700 [14,850, 44,550] | 29,700 | 59.69% | 20/32 = 0.625 |
| mvtec_ad/metal_nut | 0.05 | claimed | 412 [225, 600] | 1,089 [262, 2,630] | 0 [0, 0] | 0 | 0.39% | 3/22 = 0.136 |
| mvtec_ad/pill | 0.02 | claimed | 8,771 [5,964, 11,654] | 4,682 [2,245, 7,496] | 8,571 [5,714, 11,429] | 8,571 | 0.19% | 0/26 = 0.000 |
| mvtec_ad/screw | 0.02 | claimed | 4,390 [2,044, 7,157] | 4,606 [2,640, 6,663] | 4,000 [1,714, 6,857] | 4,000 | 0.37% | 1/41 = 0.024 |
| mvtec_ad/tile | 0.05 | claimed | 1,183 [219, 2,938] | 1,183 [219, 2,938] | 833 [0, 2,500] | 833 | 0.33% | 0/33 = 0.000 |
| mvtec_ad/toothbrush | 0.10 | not claimed (#17) | 2,302 [583, 5,272] | 2,302 [583, 5,272] | 12,375 [0, 37,125] | 12,375 | 25.53% | 4/12 = 0.333 |
| mvtec_ad/transistor | 0.05 | claimed | 1,842 [0, 5,175] | 262 [0, 525] | 1,667 [0, 5,000] | 1,667 | 0.17% | 1/60 = 0.017 |
| mvtec_ad/wood | 0.02 | claimed | 1,671 [309, 4,085] | 2,536 [494, 4,701] | 1,176 [0, 3,529] | 1,176 | 0.47% | 0/19 = 0.000 |
| mvtec_ad/zipper | 0.05 | claimed | 2,327 [670, 4,187] | 2,259 [612, 4,170] | 10,456 [0, 24,750] | 10,456 | 20.36% | 4/32 = 0.125 |
| visa/cashew | 0.02 | claimed | 4,711 [2,079, 7,864] | 2,345 [816, 4,240] | 7,300 [2,000, 15,233] | 7,300 | 6.90% | 1/50 = 0.020 |
| visa/pipe_fryum | 0.02 | claimed | 14,667 [11,333, 18,000] | 2,089 [699, 3,923] | 14,667 [11,333, 18,000] | 14,667 | 0.00% | 0/50 = 0.000 |

Not evaluated: mvtec_ad2/fruit_jelly (sup_test has 1 good / 4 defective images, need >= 4 each); mvtec_ad2/vial (sup_test has 1 good / 4 defective images, need >= 4 each).

## Probability calibration on sup_test (Brier [95 % CI], ECE; at the fit prior)

| category | Platt Brier | Platt ECE | isotonic Brier | isotonic ECE |
|---|---|---|---|---|
| bottle | 0.012 [0.002, 0.030] | 0.058 | 0.003 [0.000, 0.009] | 0.011 |
| cable | 0.035 [0.011, 0.065] | 0.062 | 0.042 [0.002, 0.093] | 0.042 |
| capsule | 0.040 [0.015, 0.072] | 0.105 | 0.052 [0.003, 0.121] | 0.058 |
| carpet | 0.058 [0.016, 0.115] | 0.067 | 0.060 [0.000, 0.146] | 0.067 |
| grid | 0.069 [0.016, 0.153] | 0.123 | 0.047 [0.002, 0.136] | 0.068 |
| hazelnut | 0.002 [0.001, 0.005] | 0.033 | 0.000 [0.000, 0.000] | 0.000 |
| leather | 0.002 [0.001, 0.004] | 0.029 | 0.000 [0.000, 0.000] | 0.000 |
| metal_nut | 0.018 [0.003, 0.044] | 0.061 | 0.022 [0.000, 0.067] | 0.025 |
| pill | 0.086 [0.038, 0.142] | 0.045 | 0.087 [0.035, 0.150] | 0.070 |
| screw | 0.099 [0.049, 0.158] | 0.149 | 0.106 [0.051, 0.170] | 0.088 |
| tile | 0.020 [0.004, 0.041] | 0.072 | 0.012 [0.000, 0.030] | 0.029 |
| toothbrush | 0.101 [0.040, 0.187] | 0.161 | 0.064 [0.004, 0.178] | 0.033 |
| transistor | 0.027 [0.003, 0.069] | 0.080 | 0.034 [0.000, 0.101] | 0.036 |
| wood | 0.047 [0.008, 0.097] | 0.075 | 0.065 [0.005, 0.152] | 0.068 |
| zipper | 0.048 [0.008, 0.101] | 0.075 | 0.049 [0.003, 0.113] | 0.052 |
| cashew | 0.072 [0.032, 0.120] | 0.109 | 0.069 [0.020, 0.129] | 0.051 |
| pipe_fryum | 0.059 [0.028, 0.100] | 0.111 | 0.058 [0.019, 0.111] | 0.080 |

## Sensitivity (costs sha256:e860bf58b48c)

Pooled saving of the guaranteed policy vs fixed_conformal over cost ratios (C_false_reject = €5 fixed): `figures/costs_sensitivity.svg`; over the defect rate: `figures/costs_defect_rate.svg`.

| C_esc/C_fr | C_rev/C_fr | guaranteed € | unconstrained € | fixed q̂ € | saving | review share |
|---|---|---|---|---|---|---|
| 0.4 | 0.001 | 141 | 57 | 5,216 | 97% | 11.19% |
| 0.4 | 0.01 | 191 | 68 | 5,216 | 96% | 11.17% |
| 0.4 | 0.1 | 690 | 123 | 5,216 | 87% | 11.11% |
| 0.4 | 1 | 5,216 | 152 | 5,216 | 0% | 0.00% |
| 0.4 | 10 | 5,216 | 152 | 5,216 | 0% | 0.00% |
| 1 | 0.001 | 188 | 136 | 5,252 | 96% | 11.16% |
| 1 | 0.01 | 238 | 159 | 5,252 | 95% | 11.15% |
| 1 | 0.1 | 735 | 211 | 5,252 | 86% | 11.09% |
| 1 | 1 | 5,252 | 333 | 5,252 | 0% | 0.00% |
| 1 | 10 | 5,252 | 333 | 5,252 | 0% | 0.00% |
| 4 | 0.001 | 415 | 270 | 5,428 | 92% | 11.08% |
| 4 | 0.01 | 464 | 332 | 5,428 | 91% | 11.08% |
| 4 | 0.1 | 956 | 647 | 5,428 | 82% | 11.04% |
| 4 | 1 | 5,428 | 867 | 5,428 | 0% | 0.00% |
| 4 | 10 | 5,428 | 867 | 5,428 | 0% | 0.00% |
| 10 | 0.001 | 855 | 464 | 5,781 | 85% | 11.01% |
| 10 | 0.01 | 903 | 528 | 5,781 | 84% | 11.01% |
| 10 | 0.1 | 1,392 | 1,060 | 5,781 | 76% | 10.99% |
| 10 | 1 | 5,781 | 1,562 | 5,781 | 0% | 0.00% |
| 10 | 10 | 5,781 | 1,562 | 5,781 | 0% | 0.00% |
| 40 | 0.001 | 2,889 | 1,141 | 7,547 | 62% | 10.81% |
| 40 | 0.01 | 2,934 | 1,228 | 7,547 | 61% | 10.81% |
| 40 | 0.1 | 3,405 | 1,903 | 7,547 | 55% | 10.79% |
| 40 | 1 | 7,547 | 4,767 | 7,547 | 0% | 0.00% |
| 40 | 10 | 7,547 | 4,767 | 7,547 | 0% | 0.00% |
| 100 | 0.001 | 6,763 | 1,903 | 11,079 | 39% | 10.69% |
| 100 | 0.01 | 6,812 | 2,028 | 11,079 | 39% | 10.69% |
| 100 | 0.1 | 7,234 | 3,416 | 11,079 | 35% | 10.67% |
| 100 | 1 | 11,079 | 8,632 | 11,079 | 0% | 0.00% |
| 100 | 10 | 11,079 | 8,632 | 11,079 | 0% | 0.00% |
| 400 | 0.001 | 25,413 | 3,737 | 28,739 | 12% | 9.95% |
| 400 | 0.01 | 25,907 | 4,626 | 28,739 | 10% | 8.98% |
| 400 | 0.1 | 26,236 | 7,285 | 28,739 | 9% | 8.97% |
| 400 | 1 | 28,739 | 12,741 | 28,739 | 0% | 0.00% |
| 400 | 10 | 28,739 | 12,741 | 28,739 | 0% | 0.00% |
| 1000 | 0.001 | 62,095 | 6,780 | 64,057 | 3% | 8.53% |
| 1000 | 0.01 | 62,133 | 7,212 | 64,057 | 3% | 8.53% |
| 1000 | 0.1 | 62,209 | 11,027 | 64,057 | 3% | 8.52% |
| 1000 | 1 | 64,057 | 21,704 | 64,057 | 0% | 0.00% |
| 1000 | 10 | 64,057 | 21,704 | 64,057 | 0% | 0.00% |
| 4000 | 0.001 | 238,560 | 10,747 | 240,652 | 1% | 7.00% |
| 4000 | 0.01 | 238,591 | 11,145 | 240,652 | 1% | 7.00% |
| 4000 | 0.1 | 238,906 | 15,783 | 240,652 | 1% | 7.00% |
| 4000 | 1 | 240,652 | 37,848 | 240,652 | 0% | 0.00% |
| 4000 | 10 | 240,652 | 37,848 | 240,652 | 0% | 0.00% |

| defect rate | guaranteed € | unconstrained € | fixed q̂ € | step-1 € | review share |
|---|---|---|---|---|---|
| 0.1% | 937 | 621 | 5,475 | 5,482 | 10.55% |
| 0.2% | 1,233 | 923 | 5,706 | 5,720 | 10.60% |
| 0.5% | 2,076 | 1,386 | 6,396 | 6,432 | 10.70% |
| 1.0% | 3,405 | 1,903 | 7,547 | 7,619 | 10.79% |
| 2.0% | 5,983 | 3,012 | 9,849 | 9,992 | 10.91% |
| 5.0% | 13,379 | 4,823 | 16,756 | 17,112 | 10.93% |

α sensitivity (pooled over the 9 categories with n_calib ≥ 49, #17 categories included; the served α stays step 2's):

| α | guaranteed € | fixed q̂ € | review share | flag rate (all good test) |
|---|---|---|---|---|
| 0.02 | 5,068 | 11,148 | 15.02% | 0.104 |
| 0.05 | 3,659 | 15,317 | 28.05% | 0.182 |
| 0.10 | 3,365 | 19,049 | 37.35% | 0.264 |
