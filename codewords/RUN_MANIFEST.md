# Harvesting Run Manifest (`TII-245`, `TII-228`, `TII-237`, `TII-235`)

## 1. `TII-235` ($n=502, m=9, t=16, mt=144, n'=425, w=33$) — Modal 10× NVIDIA L40S
- Kernel: `src/minwords_gpu_v4.cu` compiled with `-O3 -arch=sm_89 --use_fast_math -DG_GROUPS=8 -DBLOCKS_PER_GROUP=18` (`71.4 MB < 100.7 MB L2`, `144` cooperative blocks across `142` SMs).
- 10 independent workers (`seeds = 23511009..23520090`, target `4,120` each), all `rc=0`, **`out_overflows=0`**:
  - **Total exact ISD iterations**: `1,666,498,560`
  - **Total L40S GPU-hours**: `39.62` (`142,625.9 s`), **Wall-clock**: `4.30 h`
  - **Per-worker rate**: `10,934–15,593 it/s` (mean `11,684 it/s`, two Modal host boost-clock bins)
  - **Raw → Deduplicated Unique**: `41,200 → 40,353` (`2.06%` cross-worker overlap)
  - **SageMath 10.5 Lift**: `17.97 s` (`15.06 s` elimination, quotient dim `10`, direct lift dim `9`), `check_solution_track2.sage 235 == 1` (`3.37 s`)
  - **Estimated Modal cost**: `~$79 USD`

## 2. `TII-228` ($n=500, m=9, t=14, mt=126, n'=395, w=29$) — Modal 8× NVIDIA L40S
- Kernel: `src/minwords_gpu_v4.cu` (`G=8, BLOCKS_PER_GROUP=18`).
- 8 independent workers (`seeds = 22821009..22828072`, target `4,720` each), all `rc=0`, **`out_overflows=0`**:
  - **Total exact ISD iterations**: `93,574,144`
  - **Total L40S GPU-hours**: `2.15` (`7,745.3 s`), **Wall-clock**: `18.90 min`
  - **Per-worker rate**: `12,025–12,125 it/s` (mean `12,081 it/s`)
  - **Raw → Deduplicated Unique**: `37,760 → 37,084` (`1.79%` cross-worker overlap)
  - **SageMath 10.5 Lift**: `24.64 s` (`19.94 s` elimination, quotient dim `10`, direct lift dim `9`), `check_solution_track2.sage 228 == 1` (`5.43 s`)
  - **Estimated Modal cost**: `~$4.30 USD`

## 3. `TII-237` ($n=500, m=9, t=15, mt=135, n'=378, w=31$) — Colab L4 Workers
| Worker ID | GPU | Accounting Type | Runtime (GPU-h) | ISD Iterations | Rate (it/s) | Unique Words |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `tii-l4-2` | NVIDIA L4 | Measured counter | 11.19 | 381,330,944 | 9,462 | 17,009 |
| `tii-l4-b` | NVIDIA L4 | Measured counter | 5.20 | 177,663,488 | 9,485 | 8,570 |
| `tii-gpu` | NVIDIA L4 | Measured counter | 2.12 | 65,601,536 | 8,615 | 3,355 |
| `tii-237-w3` | NVIDIA L4 | Measured counter | 2.00 | 67,732,736 | 9,402 | 3,316 |
| `tii-l4-1` | NVIDIA L4 | Measured counter | 0.49 | 16,240,128 | 9,124 | 838 (+938 pre-reset) |
| `tii-237-w1/w2/w4` | NVIDIA L4 | Throughput-est. | ~1.1 | ~30M | 7.6–9.4k | 1,281 raw |
| **Total Merged** | **Colab L4** | **Combined** | **~22.1** | **~7.4×10⁸** | — | **29,742** |

- **Lincoln–Petersen Cross-Check** (`tii-l4-2` vs `tii-l4-b`): `|A|=17,009`, `|B|=8,570`, `|A∩B|=2,215` → `N_hat = 65,809 ± 1,123` (SE), 95% CI `[63,608, 68,010]`.
- **Total Colab Account Spend** (development + v1–v4 profiling + A100 benchmark + `TII-245` + `TII-237`): `68.97 CU` (`946.08 → 877.11 CU`).

## 4. `TII-245` ($n=493, m=9, t=12, mt=108, n'=326, w=25$) — Colab 1× NVIDIA L4
- **Exact ISD iterations**: `12,958,464` in `1,175.5 s` (`19.6 min`, `11,024 it/s` average, `11,035 it/s` peak), `24,500` unique verified weight-25 codewords.
- **SageMath 10.7 Lift**: `9.78 s`, `check_solution_track2.sage 245 == 1`.
