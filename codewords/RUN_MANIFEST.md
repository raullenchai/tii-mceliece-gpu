# TII-237 Harvesting Run Manifest

This ledger documents the individual Colab GPU worker runs merged into
`codewords/words_237_np378.txt.gz` (29,742 unique host-verified weight-31 codewords
of `H_A = H_pub[:, :378]`, `rank(H_A) = 135`).

| Worker ID | GPU | Accounting Type | Runtime (GPU-h) | ISD Iterations | Rate (it/s) | Unique Words | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `tii-l4-2` | NVIDIA L4 | Measured counter | 11.19 | 381,330,944 | 9,462 | 17,009 | Primary L4 worker (40,300 s) |
| `tii-l4-b` | NVIDIA L4 | Measured counter | 5.20 | 177,663,488 | 9,485 | 8,570 | Secondary L4 worker (18,731 s) |
| `tii-gpu` | NVIDIA A100-40GB | Measured counter | 2.12 | 65,601,536 | 8,615* | 3,355 | Earlier A100 run (7,615 s; *mixed v3→v4) |
| `tii-237-w3` | NVIDIA L4 | Measured counter | 2.00 | 67,732,736 | 9,402 | 3,316 | Earlier L4 run (7,204 s) |
| `tii-l4-1` | NVIDIA L4 | Measured counter | 0.49 | 16,240,128 | 9,122 | 838 | Earlier L4 run (1,780 s) + pre-reset merge (+938 net unique) |
| `tii-237-w1/w2/w4` | NVIDIA L4 | Throughput-est. | ~1.1 | ~30M | ~7.6–9.4k | 1,281 raw | Short exploratory runs (`w2`/`w3` overlapped via pre-fix `| 1ULL` bug) |
| **Total Merged** | **Colab L4/A100** | **Combined** | **~22.1** | **~7.0×10⁸** | — | **29,742** | Deduplicated union in `words_237_np378.txt.gz` |

- **Final Lincoln–Petersen Cross-Check** (`tii-l4-2` vs `tii-l4-b`):
  `|A| = 17,009`, `|B| = 8,570`, `|A ∩ B| = 2,215` → heuristic `N_hat = 65,809 ± 1,123` (SE), 95% CI `[63,608, 68,010]`.
- **Total Colab Account Spend** (including all v1–v4 development, profiling, TII-245 control verification, TII-237 harvesting, and SageMath algebraic recovery): **68.97 CU** (`946.08 CU` → `877.11 CU`; L4 rate `1.94 CU/hr` as of November 2026).
