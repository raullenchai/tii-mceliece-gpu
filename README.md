# Packing Stern into L2: GPU-Accelerated Primal-Shortened ISD for High-Codimension Binary Goppa Key Recovery

**[IACR ePrint 2026/2993](https://eprint.iacr.org/2026/2993)** · **[Interactive Visual Explainer](https://raullenchai.github.io/tii-mceliece-gpu/explainer.html)** · **[Manuscript PDF](paper/main.pdf)** · **[Run Manifest](codewords/RUN_MANIFEST.md)**

> *Also archived in our unified open research collection at [raullenchai/open-research](https://github.com/raullenchai/open-research). This standalone repository is preserved permanently so links in IACR ePrint 2026/2993 remain stable.*

Accompanying artifact repository for **Raullen Chai & Xinxin Fan** (MachineFi Labs), *"Packing Stern into L2: GPU-Accelerated Primal-Shortened ISD for High-Codimension Binary Goppa Key Recovery"* (`paper/main.pdf`).

This repository contains our CUDA-accelerated **L2-Sized Cooperative Stern-Style ISD** samplers (`src/minwords_gpu_v4.cu` for lossless $mt \le 145$ and `src/minwords_gpu_v5.cu` for truncated 16B range pre-filtering up to $mt \le 192$), baseline evolution (`src/baselines/`), manuscript (`paper/main.pdf`), all **174,796** verified harvested codewords (`codewords/`), zero-dependency verifier (`src/verify_witnesses.py`), and five verified equivalent private-key witnesses (`witnesses/`) for Track 2 of the **TII McEliece Key Recovery Challenges**.

## Attribution & Scope

This work builds directly on **MiniMax's Primal-Shortened ISD** algebraic recovery framework and Mode-6 Stern-style sampler (`minwords_mt.c`, $p=3+3, l=17$, 16 KB occupancy bitmap), released under Apache-2.0 on November 20, 2026 ([MiniMax-AI/Research](https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece)).

Our GPU-oriented modifications (`src/minwords_gpu_v4.cu`, `src/minwords_gpu_v5.cu`, and shortening-window optimization):

1. **16-Byte `ulonglong2` Zero-Index Packing & Truncated Range Pre-Filtering**: For $mt \le 145$ (`v4`), all $mt - 17 \le 128$ remaining syndrome bits pack losslessly into one `ulonglong2` (4 slots/bucket = 64-byte aligned payload), omitting left-triple indices from the hot bucket table. For $145 < mt \le 192$ (`v5`), we keep the identical 16-byte bucket layout, apply the lossless range pre-filter $(w-6)-(mt-145) \le \operatorname{popc}_{128} \le w-6$ (pass prob $2^{-32.28}$ on `TII-185`), and verify exact $mt$-bit weight upon rare hits.
2. **Cooperative L2-Sized Execution ($G=4$ on L4 / $G=8$ on L40S)**: All SMs cooperate on $G=4$ tables (`35.7 MB < 50.3 MB` L4 L2) or $G=8$ tables (`71.4 MB < 100.7 MB` L40S L2 via `-DG_GROUPS=8 -DBLOCKS_PER_GROUP=18`), replacing `1.04–1.49 GB` per-SM tables.
3. **Batched Full-SM Systematic Reduction**: Batching $64G$ independent random systematic reductions per launch (`256` on L4, `512` on L40S) removes the `74.0%` Gaussian-elimination bottleneck (`v3` $\to$ `v4`: `3.5×` on `TII-237`, `3.17×` on `TII-245`).
4. **GPU-Aware Optimal Shortening Window $n'$**: Widening $n'$ beyond narrow CPU-oriented choices (`TII-228`: `355 → 395`; `TII-235`: `393 → 425`; `TII-185`: `447`) reduces modeled GPU-hours by `34%–47%` under a Poisson-bucket and coupon-collector model while keeping SageMath M4RI lifting under `40 s`.

## Verified Results ($mt = 108 \to 153$)

| Challenge | $(n, m, t)$ | $mt$ | $n'$ | Prior / Reference | Our Hardware & Timing | Algebraic Lift | Official Check |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **`TII-245`** (control) | $(493, 9, 12)$ | `108` | `326` | MiniMax (64-core Xeon): `2,510 s` e2e (`44.6 ch`) | **1× Colab L4**: `1,175.5 s` ISD (`12.96M` iters, `24,500` words; `~1.91×` normalized) | `9.78 s` | **PASS (`1`)** |
| **`TII-228`** (head-to-head) | $(500, 9, 14)$ | `126` | `395` | MiniMax (`2,048–5,120` EPYC): `57.1 min` (`2,463 ch`) | **8× Modal L40S**: **`18.9 min` wall** (`2.15 GPU-h`, `93.57M` iters, `37,084` words, `~$4.30`) | `24.64 s` | **PASS (`1`)** |
| **`TII-237`** (new) | $(500, 9, 15)$ | `135` | `378` | *Unsolved publicly* (prior max $mt=126$) | **Colab L4**: `~22.1 GPU-h` (`~7.4×10⁸` iters, `29,742` words) | `23.23 s` | **PASS (`1`)** |
| **`TII-235`** (new, 128b limit) | $(502, 9, 16)$ | `144` | `425` | *Unsolved publicly* (10 punctures) | **10× Modal L40S**: **`4.30 h` wall** (`39.62 GPU-h`, `1.67×10⁹` iters, `40,353` words, `~$79`) | **`17.97 s`** | **PASS (`1`)** |
| **`TII-185`** (new, >128b `v5`) | $(511, 9, 17)$ | **`153`** | `447` | *Unsolved publicly* (past 128b ceiling) | **7× L40S + 3× L4 (`v5`)**: **`21.1 h` wall** (`140.4 L40S-h + 60 L4-h`, `6.38×10⁹` iters, `43,117` words) | **`39.01 s`** | **PASS (`1`)** |

*Security Note:* Classic McEliece (NIST PQC Round-4 candidate; NIST selected HQC in March 2025) uses $m \ge 12, t \ge 64, mt \ge 768$. These experiments do not demonstrate practical key recovery against proposed Classic McEliece parameter sets.

## Independent Verification (Zero Dependencies)

Run from the repository root (downloads official public keys on first run into the OS temp directory, or pass `--pk-dir <dir>` for offline verification; enforces hardcoded SHA-256 digests):

```bash
python3 src/verify_witnesses.py --check-words
```

Or verify directly with the unmodified official [ElenaKirshanova/tii_decoding_challenge](https://github.com/ElenaKirshanova/tii_decoding_challenge) SageMath checker:

```bash
git clone https://github.com/ElenaKirshanova/tii_decoding_challenge.git /tmp/tii_decoding_challenge
cd /tmp/tii_decoding_challenge
for sec in 245 228 237 235 185; do
  sage check_solution_track2.sage $sec /path/to/tii-mceliece-gpu/witnesses/sk_McEliece_${sec}.txt
done
```

## Reproducing the Algebraic Lift (Seconds, SageMath + MiniMax Repo)

```bash
git clone https://github.com/MiniMax-AI/Research.git /tmp/minimax_research
cd /tmp/minimax_research/Cryptography/TII-McEliece/artifacts && bash fetch_third_party.sh
sage -python /path/to/tii-mceliece-gpu/src/solve_from_words.py \
  235 425 /path/to/tii-mceliece-gpu/codewords/words_235_np425.txt.gz \
  /tmp/minimax_research/Cryptography/TII-McEliece/artifacts/goppa_key_recovery /tmp/tii_out
```

## Building & Running the GPU Harvester

```bash
# 1. Export shortened parity-check matrix H_A in LSB-first hex format
python3 src/export_ha.py 235 425 /tmp/ha_235.txt

# 2a. Build for L4 (50.3 MB L2, G=4) and A100 (41.9 MB L2, G=4)
nvcc -O3 --use_fast_math \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  src/minwords_gpu_v4.cu -o minwords_gpu_v4

# 2b. Build for L40S (100.7 MB L2, 142 SMs, G=8, 144 blocks)
nvcc -O3 --use_fast_math -arch=sm_89 -DG_GROUPS=8 -DBLOCKS_PER_GROUP=18 \
  src/minwords_gpu_v4.cu -o minwords_gpu_v4_l40s

# 3. Run harvester
MINWORDS_CKPT=/tmp/words_235_w1.txt ./minwords_gpu_v4_l40s /tmp/ha_235.txt 144 425 4120 33 6 23511009 36000 > /dev/null
```

## Directory Layout

```text
tii-mceliece-gpu/
├── LICENSE                        # Apache License 2.0
├── NOTICE                         # Upstream MiniMax & TII attribution notices
├── README.md
├── explainer.html                 # Interactive English visual walkthrough
├── codewords/
│   ├── RUN_MANIFEST.md            # Per-worker GPU-hour & iteration ledger
│   ├── words_245_np326.txt.gz     # 24,500 verified weight-25 codewords (TII-245)
│   ├── words_228_np395.txt.gz     # 37,084 verified weight-29 codewords (TII-228)
│   ├── words_237_np378.txt.gz     # 29,742 verified weight-31 codewords (TII-237)
│   ├── words_235_np425.txt.gz     # 40,353 verified weight-33 codewords (TII-235)
│   └── words_185_np447.txt.gz     # 43,117 verified weight-35 codewords (TII-185)
├── paper/
│   ├── main.tex                   # IACR ePrint manuscript (with TikZ figures)
│   ├── main.pdf                   # Compiled PDF
│   └── eprint_submission.txt      # Ready-to-paste IACR ePrint form metadata
├── src/
│   ├── minwords_gpu_v4.cu         # L2-sized cooperative CUDA Stern ISD (mt <= 145, Apache-2.0)
│   ├── minwords_gpu_v5.cu         # Truncated 16B range pre-filter extension (mt <= 192, Apache-2.0)
│   ├── export_ha.py               # Standalone H_A hex exporter from official TII public keys
│   ├── solve_from_words.py        # SageMath algebraic recovery & official TII checker wrapper
│   ├── verify_witnesses.py        # Zero-dependency pure-Python verifier (all 5 keys + 174,796 words)
│   └── baselines/                 # Historical v1-v3 CUDA kernels
└── witnesses/
    ├── sk_McEliece_245.txt        # Verified equivalent private key for TII-245 (check_key == 1)
    ├── sk_McEliece_228.txt        # Verified equivalent private key for TII-228 (check_key == 1)
    ├── sk_McEliece_237.txt        # Verified equivalent private key for TII-237 (check_key == 1)
    ├── sk_McEliece_235.txt        # Verified equivalent private key for TII-235 (check_key == 1)
    └── sk_McEliece_185.txt        # Verified equivalent private key for TII-185 (check_key == 1)
```
