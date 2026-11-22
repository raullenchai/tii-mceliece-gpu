# Packing Stern into L2: GPU-Accelerated Primal-Shortened ISD & Key Recovery on `TII-237` ($mt=135$)

Accompanying artifact repository for **Raullen Qi Chai** (MachineFi Labs), *"Packing Stern into L2: GPU-Accelerated Primal-Shortened ISD and Key Recovery on TII-237 ($mt=135$)"* (`paper/main.pdf`).

This repository contains our CUDA-accelerated **L2-Sized Cooperative Stern-Style ISD** sampler (`src/minwords_gpu_v4.cu`), baseline evolution (`src/baselines/`), manuscript (`paper/main.pdf`), harvested codeword datasets (`codewords/`), zero-dependency verifier (`src/verify_witnesses.py`), and verified equivalent private-key witnesses (`witnesses/`) for Track 2 of the **TII McEliece Key Recovery Challenges**.

## Attribution & Scope

This work builds directly on **MiniMax's Primal-Shortened ISD** algebraic recovery framework, Mode-6 Stern-style sampler (`minwords_mt.c`, $p=3+3, l=17$, 16 KB occupancy bitmap), and shortening schedule (`optimal_shortening_sizes`), released under Apache-2.0 on November 20, 2026 ([MiniMax-AI/Research](https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece)).

Our contribution is focused on **GPU memory-hierarchy engineering** of the minimum odd-weight codeword harvester (`src/minwords_gpu_v4.cu`):

1. **Lossless 16-Byte `ulonglong2` Syndrome Packing (Zero Index Storage)**: For $mt \le 145$ and $l = 17$, all $mt - 17 \le 128$ remaining syndrome bits pack losslessly into one `ulonglong2` (4 slots/bucket = 64-byte aligned payload, plus separate 4-byte count array), omitting left triple indices from the hot bucket table.
2. **Cooperative L2-Sized Execution ($G = 4$)**: Rather than one independent hash table per SM (`1.04–1.49 GB`), all SMs cooperate on $G = 4$ concurrent reductions (`35,717,120 B` / `35.7 MB` working set), sized to fit below nominal NVIDIA L4 (`50.3 MB`) and A100 (`41.9 MB`) L2 caches.
3. **Full-SM Batched Systematic Reduction**: CUDA event profiling of `v3` showed sequential per-group Gaussian elimination (`k_sys`) consumed **74.0%** of kernel time (`68.0 ms / 91.9 ms`). Batching $256$ independent reductions in `k_sys<<<256, 256>>>` boosted L4 throughput $3.5\times$ on `TII-237` (from `2,671` to **`9,462 it/s`**) and $3.17\times$ on `TII-245` (**`11,035 it/s`**).

## Results

| Challenge | $(n, m, t)$ | $mt$ | Reference / Prior Status | Our Compute & Timing | Algebraic Lift | Verification |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **`TII-245`** (control) | $(493, 9, 12)$ | `108` | MiniMax (64-core Xeon): `2,500 s` ISD (`26,352` words), `2,510 s` e2e | **1× L4**: `1,175.5 s` ISD (`24,500` words, `12.96M` iters), `1,185.3 s` e2e (`2.12×` e2e, `~1.91×` normalized) | `9.78 s` | **PASS** (`1`) |
| **`TII-237`** | $(500, 9, 15)$ | **`135`** | Highest publicly documented Track-2 $mt$ known to us as of 2026-11-22 (prior highest: `TII-228`, $mt=126$, `2,463` core-h on `2,048–5,120` EPYC cores) | **Colab L4 GPUs**: `~22.1 GPU-h` (`~7.4×10⁸` iters, `29,742` unique words; `68.97 CU` total account spend across dev + runs) | **`23.23 s`** | **PASS** (`1`) |

### Cryptographic Digests ($\mathbb{F}_{2^9} = \mathbb{F}_2[a]/(a^9 + a^4 + 1)$)

- **`TII-245`**:
  - `pk_McEliece_245.txt` SHA-256: `ea029e86d34d13334bb9439f87333a8e87ebd902379ec515331d2bed78e973a8`
  - `sk_McEliece_245.txt` SHA-256: `967d2f4e934ca7617bdd71cfb5a0fcd98ec0f54a2c50db5170899ab3debac523`
  - `codewords/words_245_np326.txt.gz` SHA-256: `b54b11b6c695b32a08aedb545cfb68dac9e0e3e7e1184175474344bfc59b4a11`
- **`TII-237`**:
  - `pk_McEliece_237.txt` SHA-256: `b3cba630833e00d8e225d1943be87c31ddd74f270bd73f61c373aa70b1356f3b`
  - `sk_McEliece_237.txt` SHA-256: `128e26137b2b2e8128316cbc5e1c7766073d790e26a308a9766a120f05770e0f`
  - `codewords/words_237_np378.txt.gz` SHA-256: `7793b74438d76235c8f5048a6b5644113b8372d999f837e1c09c274b2a472531`

*Security Note:* Classic McEliece (a NIST PQC Round-3 finalist / Round-4 candidate; NIST selected HQC in March 2025) uses parameter sets with $m \ge 12, t \ge 64, mt \ge 768$. These experiments do not demonstrate practical key recovery against proposed Classic McEliece parameters.

## Independent Verification (Recommended: Zero Dependencies)

Run from inside `tii_mceliece_gpu/` (fetches official public keys on first run into the OS temp directory, or pass `--pk-dir <dir>` for offline verification; enforces hardcoded SHA-256 digests):

```bash
# Pure-Python verification of both equivalent private keys AND all 24,500 + 29,742 harvested codewords
python3 src/verify_witnesses.py --check-words
```

Or verify directly with the unmodified official [ElenaKirshanova/tii_decoding_challenge](https://github.com/ElenaKirshanova/tii_decoding_challenge) SageMath checker (requires SageMath):

```bash
git clone https://github.com/ElenaKirshanova/tii_decoding_challenge.git /tmp/tii_decoding_challenge
cd /tmp/tii_decoding_challenge
sage check_solution_track2.sage 245 /path/to/tii_mceliece_gpu/witnesses/sk_McEliece_245.txt   # -> 1
sage check_solution_track2.sage 237 /path/to/tii_mceliece_gpu/witnesses/sk_McEliece_237.txt   # -> 1
```

## Reproducing the Algebraic Lift (Seconds, Requires SageMath + MiniMax Repo)

```bash
git clone https://github.com/MiniMax-AI/Research.git /tmp/minimax_research
cd /tmp/minimax_research/Cryptography/TII-McEliece/artifacts && bash fetch_third_party.sh
sage -python /path/to/tii_mceliece_gpu/src/solve_from_words.py \
  237 378 /path/to/tii_mceliece_gpu/codewords/words_237_np378.txt.gz \
  /tmp/minimax_research/Cryptography/TII-McEliece/artifacts/goppa_key_recovery /tmp/tii_out
```
*(Note: `solve_from_words.py` automatically downloads and SHA-256-verifies `pk_McEliece_237.txt` into `tii_repo/public_keyRec/` if absent, since MiniMax's `fetch_third_party.sh` only extracts its 94 solved keys.)*

## Building & Running the GPU Harvester

`src/minwords_gpu_v4.cu` supports `mode=6` ($p=3+3, l=17$), $mt \le 145$, and $k' = n' - mt \le 320$, with strict host-side rank/input validation, exact parity verification ($H_A \cdot v = 0$), and incremental checkpointing via `MINWORDS_CKPT`:

```bash
# 1. Export shortened parity-check matrix H_A in LSB-first hex format
python3 src/export_ha.py 237 378 /tmp/ha_237.txt

# 2. Build fat binary targeting both A100 (sm_80) and L4 (sm_89)
nvcc -O3 --use_fast_math \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  src/minwords_gpu_v4.cu -o minwords_gpu_v4

# 3. Run harvester (use distinct seeds across parallel workers)
MINWORDS_CKPT=/tmp/words_237_w1.txt ./minwords_gpu_v4 /tmp/ha_237.txt 135 378 30000 31 6 237101 43200 > /dev/null
```

## Directory Layout

```text
tii_mceliece_gpu/
├── LICENSE                        # Apache License 2.0
├── NOTICE                         # Upstream MiniMax & TII attribution notices
├── README.md
├── explainer.html                 # Interactive Chinese visual walkthrough
├── codewords/
│   ├── RUN_MANIFEST.md            # Per-worker TII-237 GPU-hour & iteration ledger
│   ├── words_245_np326.txt.gz     # 24,500 verified weight-25 codewords for TII-245
│   └── words_237_np378.txt.gz     # 29,742 verified weight-31 codewords for TII-237
├── paper/
│   ├── main.tex                   # IACR ePrint-ready manuscript
│   └── main.pdf                   # Compiled PDF
├── src/
│   ├── minwords_gpu_v4.cu         # L2-sized cooperative CUDA Stern-style ISD (Apache-2.0 derived)
│   ├── export_ha.py               # Standalone H_A hex exporter from official TII public keys
│   ├── solve_from_words.py        # SageMath algebraic recovery & official TII checker wrapper
│   ├── verify_witnesses.py        # Zero-dependency pure-Python witness & codeword verifier
│   └── baselines/                 # Historical v1-v3 CUDA kernels from Table 1
└── witnesses/
    ├── sk_McEliece_245.txt        # Verified equivalent private key for TII-245 (check_key == 1)
    └── sk_McEliece_237.txt        # Verified equivalent private key for TII-237 (check_key == 1)
```
