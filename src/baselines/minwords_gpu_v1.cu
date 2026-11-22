// Historical baseline CUDA kernel for Table 1 ablation.
// Derived from MiniMax minwords_mt.c (Copyright (C) 2026 MiniMax, Apache License 2.0,
// https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece).
// See ../../LICENSE and ../../NOTICE for full license and modification details.
//
// GPU Bit-Parallel ISD Sampler for Minimum-Weight Codewords of ker(H)
// Drop-in CUDA replacement for MiniMax's minwords_mt (Route A Primal-Shortened ISD)
// Optimized for NVIDIA A100 (sm_80), compatible with T4/V100/L4/H100.
//
// Supports:
//   - Mode 3: Lee-Brickell p=3 (for small/medium codes)
//   - Mode 6: Shared-Memory 16KB Bitmap Pre-Filtered Stern-Dumer (p = 3+3 = 6, l = 17, 6 slots/bkt)
//   - Mode 0: Auto (Mode 6 when rows >= 88 and weight >= 21, else Mode 3)
//   - Incremental checkpoint/resume via MINWORDS_CKPT environment variable
//   - Host-side exact parity verification (H * v == 0 and wt(v) == weight) on every emitted word
//
// Build (A100):
//   nvcc -O3 -arch=sm_80 minwords_gpu.cu -o minwords_gpu
//
// Usage (100% drop-in compatible with minwords_mt):
//   minwords_gpu <matrix-file> <rows> <cols> <target> <weight> <mode> <seed> [max_seconds] [unused_threads]

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 192
#define MAXC 512
#define MAXNW (MAXC / 64)
#define SW 3 /* up to 192 syndrome bits (3 x 64-bit words) */

#define S6_BITS 17
#define S6_BKTS (1u << S6_BITS) /* 131,072 buckets */
#define S6_MASK (S6_BKTS - 1u)
#define S6_SLOTS 6              /* 6 entries/bucket -> <1.5% drop at mid=125 */

#define THREADS_PER_BLOCK 256
#define MAX_OUT_PER_LAUNCH 4096
#define MAX_W 64

/* Host-side dedup hash table (4M slots) */
#define HB 22
#define HM ((1u << HB) - 1u)

typedef struct __align__(32) {
    uint64_t s0;
    uint64_t s1;
    uint64_t s2;
    uint32_t c12; /* c1 | (c2 << 16) */
    uint32_t c3;  /* c3 */
} stern6_entry_t;

typedef struct {
    uint16_t idx[MAX_W];
} found_word_t;

__constant__ uint64_t c_base[MAXR][MAXNW];

static uint64_t h_base[MAXR][MAXNW];
static uint64_t h_htab[1u << HB];
static int h_found_count = 0;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__host__ __device__ static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t hash_support(const uint16_t *idx, int w) {
    uint64_t key = 1469598103934665603ULL;
    for (int i = 0; i < w; ++i) {
        key ^= (uint64_t)idx[i];
        key *= 1099511628211ULL;
    }
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key | 1ULL;
}

static int verify_codeword_host(const uint16_t *idx, int w, int rows, int cols) {
    for (int i = 0; i < w; ++i) {
        if (idx[i] >= cols) return 0;
        if (i > 0 && idx[i] <= idx[i - 1]) return 0;
    }
    for (int r = 0; r < rows; ++r) {
        int parity = 0;
        for (int i = 0; i < w; ++i) {
            uint16_t c = idx[i];
            parity ^= (int)((h_base[r][c >> 6] >> (c & 63)) & 1ULL);
        }
        if (parity != 0) return 0;
    }
    return 1;
}

static int htab_insert(uint64_t key) {
    uint32_t slot = (uint32_t)key & HM;
    while (1) {
        uint64_t cur = h_htab[slot];
        if (cur == key) return 0;
        if (cur == 0ULL) {
            h_htab[slot] = key;
            return 1;
        }
        slot = (slot + 1u) & HM;
    }
}

__device__ static inline void emit_word_gpu(
    const uint16_t *fc, int nfree,
    uint64_t s0, uint64_t s1, uint64_t s2,
    const uint16_t *s_pivots,
    int rows, int sw_used, int weight,
    found_word_t *d_out, int *d_out_cnt
) {
    int slot = atomicAdd(d_out_cnt, 1);
    if (slot >= MAX_OUT_PER_LAUNCH) return;

    uint16_t idx[MAX_W];
    int n = 0;
    for (int i = 0; i < nfree && n < MAX_W; ++i) idx[n++] = fc[i];

    uint64_t words[3] = { s0, s1, s2 };
    for (int q = 0; q < sw_used; ++q) {
        uint64_t w = words[q];
        while (w && n < MAX_W) {
            int bit = __ffsll((long long)w) - 1;
            int row = (q << 6) + bit;
            if (row < rows) idx[n++] = s_pivots[row];
            w &= (w - 1ULL);
        }
    }
    if (n != weight) {
        d_out[slot].idx[0] = 0xFFFF;
        return;
    }
    /* Insertion sort */
    for (int i = 1; i < weight; ++i) {
        uint16_t v = idx[i];
        int j = i - 1;
        while (j >= 0 && idx[j] > v) {
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = v;
    }
    for (int i = 0; i < weight; ++i) {
        d_out[slot].idx[i] = idx[i];
    }
}

/* Cooperative block-level random systematic reduction inside shared memory */
__device__ static int block_sample_systematic(
    uint64_t *rng_state,
    int rows, int cols, int nw, int sw_used,
    uint64_t s_sysrow[MAXR][MAXNW],
    uint16_t s_pivots[MAXR],
    uint16_t s_npi[MAXC],
    uint8_t s_taken[MAXC],
    uint64_t s_colmask[320][SW],
    int *s_shared_flag
) {
    const int tid = threadIdx.x;
    while (1) {
        for (int c = tid; c < cols; c += blockDim.x) s_taken[c] = 0;
        __syncthreads();

        if (tid == 0) {
            uint64_t st = *rng_state;
            for (int i = 0; i < rows; ++i) {
                int c;
                do {
                    c = (int)(splitmix64(&st) % (uint64_t)cols);
                } while (s_taken[c]);
                s_taken[c] = 1;
                s_pivots[i] = (uint16_t)c;
            }
            *rng_state = st;
        }
        __syncthreads();

        if (tid < rows) {
            for (int q = 0; q < nw; ++q) s_sysrow[tid][q] = c_base[tid][q];
        }
        __syncthreads();

        int ok = 1;
        for (int rank = 0; rank < rows; ++rank) {
            int col = s_pivots[rank];
            int w = col >> 6;
            int b = col & 63;

            if (tid == 0) {
                int p = rank;
                while (p < rows && (((s_sysrow[p][w] >> b) & 1ULL) == 0ULL)) ++p;
                *s_shared_flag = p;
            }
            __syncthreads();

            int p = *s_shared_flag;
            if (p == rows) {
                ok = 0;
                break;
            }
            if (p != rank) {
                if (tid < nw) {
                    uint64_t tmp = s_sysrow[p][tid];
                    s_sysrow[p][tid] = s_sysrow[rank][tid];
                    s_sysrow[rank][tid] = tmp;
                }
                __syncthreads();
            }
            if (tid < rows && tid != rank) {
                if ((s_sysrow[tid][w] >> b) & 1ULL) {
                    for (int q = 0; q < nw; ++q) {
                        s_sysrow[tid][q] ^= s_sysrow[rank][q];
                    }
                }
            }
            __syncthreads();
        }
        if (ok) break;
    }

    if (tid == 0) {
        int npiv = 0;
        for (int j = 0; j < cols; ++j) {
            if (!s_taken[j]) {
                s_npi[npiv++] = (uint16_t)j;
            }
        }
        *s_shared_flag = npiv;
    }
    __syncthreads();

    int npiv = *s_shared_flag;
    if (npiv > 320) npiv = 320;

    for (int j = tid; j < npiv; j += blockDim.x) {
        int col = s_npi[j];
        int w = col >> 6;
        int b = col & 63;
        uint64_t m0 = 0ULL, m1 = 0ULL, m2 = 0ULL;
        int r_lim0 = (rows < 64) ? rows : 64;
        for (int i = 0; i < r_lim0; ++i) {
            if ((s_sysrow[i][w] >> b) & 1ULL) m0 |= (1ULL << i);
        }
        if (sw_used >= 2) {
            int r_lim1 = (rows < 128) ? rows : 128;
            for (int i = 64; i < r_lim1; ++i) {
                if ((s_sysrow[i][w] >> b) & 1ULL) m1 |= (1ULL << (i - 64));
            }
        }
        if (sw_used >= 3) {
            for (int i = 128; i < rows; ++i) {
                if ((s_sysrow[i][w] >> b) & 1ULL) m2 |= (1ULL << (i - 128));
            }
        }
        s_colmask[j][0] = m0;
        s_colmask[j][1] = m1;
        s_colmask[j][2] = m2;
    }
    __syncthreads();
    return npiv;
}

/* Mode 6 GPU Kernel: Shared-Memory 16KB Bitmap + Stern-Dumer (p = 3+3 = 6, l = 17) */
__global__ void kernel_stern6(
    int rows, int cols, int nw, int sw_used, int weight,
    int iters_per_block, uint64_t launch_seed,
    uint32_t *d_bkt_cnt_all,
    stern6_entry_t *d_bkt_entries_all,
    found_word_t *d_out, int *d_out_cnt
) {
    __shared__ uint64_t s_sysrow[MAXR][MAXNW];
    __shared__ uint64_t s_colmask[320][SW];
    __shared__ uint16_t s_pivots[MAXR];
    __shared__ uint16_t s_npi[MAXC];
    __shared__ uint8_t s_taken[MAXC];
    __shared__ uint32_t s_bmap[S6_BKTS / 32]; /* 16 KB in shared memory */
    __shared__ int s_flag;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int inside = weight - 6;
    if (inside < 0) return;

    uint32_t *bkt_cnt = d_bkt_cnt_all + (size_t)bid * S6_BKTS;
    stern6_entry_t *bkt_entries = d_bkt_entries_all + (size_t)bid * S6_BKTS * S6_SLOTS;

    uint64_t rng = launch_seed ^ ((uint64_t)(bid + 1) * 0x9e3779b97f4a7c15ULL);
    splitmix64(&rng);

    for (int it = 0; it < iters_per_block; ++it) {
        int npiv = block_sample_systematic(
            &rng, rows, cols, nw, sw_used,
            s_sysrow, s_pivots, s_npi, s_taken, s_colmask, &s_flag
        );
        int mid = npiv / 2;
        if (mid < 3) continue;
        if (mid > 160) mid = 160;
        int r_len = npiv - mid;
        if (r_len < 3) continue;

        /* Zero shared bitmap and global bucket counts */
        for (uint32_t i = tid; i < (S6_BKTS / 32u); i += blockDim.x) s_bmap[i] = 0u;
        for (uint32_t i = tid; i < S6_BKTS; i += blockDim.x) bkt_cnt[i] = 0u;
        __syncthreads();

        /* Phase 1: Left triples (a < b < c < mid) */
        int n_lpairs = (mid * (mid - 1)) / 2;
        for (int pidx = tid; pidx < n_lpairs; pidx += blockDim.x) {
            /* Decode triangular index pidx -> (a, b) with 0 <= a < b < mid */
            /* Fast exact decode using float sqrt + adjustment */
            int b = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
            while ((b * (b - 1)) / 2 > pidx) --b;
            while (((b + 1) * b) / 2 <= pidx) ++b;
            int a = pidx - (b * (b - 1)) / 2;

            uint64_t zab0 = s_colmask[a][0] ^ s_colmask[b][0];
            uint64_t zab1 = s_colmask[a][1] ^ s_colmask[b][1];
            uint64_t zab2 = s_colmask[a][2] ^ s_colmask[b][2];
            uint32_t c12 = (uint32_t)s_npi[a] | ((uint32_t)s_npi[b] << 16);

            for (int c = b + 1; c < mid; ++c) {
                uint64_t s0 = zab0 ^ s_colmask[c][0];
                uint32_t h = (uint32_t)(s0 & S6_MASK);
                uint32_t slot = atomicAdd(&bkt_cnt[h], 1u);
                if (slot == 0u) {
                    atomicOr(&s_bmap[h >> 5], 1u << (h & 31u));
                }
                if (slot < S6_SLOTS) {
                    stern6_entry_t *E = &bkt_entries[(size_t)h * S6_SLOTS + slot];
                    E->s0 = s0;
                    E->s1 = zab1 ^ s_colmask[c][1];
                    E->s2 = zab2 ^ s_colmask[c][2];
                    E->c12 = c12;
                    E->c3 = (uint32_t)s_npi[c];
                }
            }
        }
        __syncthreads();

        /* Phase 2: Right triples (mid <= d < e < f < npiv) */
        int n_rpairs = (r_len * (r_len - 1)) / 2;
        if (sw_used == 2) {
            for (int pidx = tid; pidx < n_rpairs; pidx += blockDim.x) {
                int eb = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
                while ((eb * (eb - 1)) / 2 > pidx) --eb;
                while (((eb + 1) * eb) / 2 <= pidx) ++eb;
                int da = pidx - (eb * (eb - 1)) / 2;
                int d = mid + da;
                int e = mid + eb;

                uint64_t zde0 = s_colmask[d][0] ^ s_colmask[e][0];
                uint64_t zde1 = s_colmask[d][1] ^ s_colmask[e][1];
                uint16_t cd = s_npi[d], ce = s_npi[e];

                for (int f = e + 1; f < npiv; ++f) {
                    uint64_t r0 = zde0 ^ s_colmask[f][0];
                    uint32_t h = (uint32_t)(r0 & S6_MASK);
                    if ((s_bmap[h >> 5] & (1u << (h & 31u))) == 0u) continue;
                    uint32_t cnt = bkt_cnt[h];
                    if (cnt > S6_SLOTS) cnt = S6_SLOTS;
                    uint64_t r1 = zde1 ^ s_colmask[f][1];
                    const stern6_entry_t *baseE = &bkt_entries[(size_t)h * S6_SLOTS];
                    for (uint32_t k = 0; k < cnt; ++k) {
                        uint64_t s0 = baseE[k].s0 ^ r0;
                        int w0 = __popcll(s0);
                        if (w0 <= inside) {
                            uint64_t s1 = baseE[k].s1 ^ r1;
                            if (w0 + __popcll(s1) == inside) {
                                uint32_t c12 = baseE[k].c12;
                                uint16_t fc[6] = {
                                    (uint16_t)(c12 & 0xFFFFu), (uint16_t)(c12 >> 16),
                                    (uint16_t)baseE[k].c3, cd, ce, s_npi[f]
                                };
                                emit_word_gpu(fc, 6, s0, s1, 0ULL, s_pivots, rows, 2, weight, d_out, d_out_cnt);
                            }
                        }
                    }
                }
            }
        } else {
            for (int pidx = tid; pidx < n_rpairs; pidx += blockDim.x) {
                int eb = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
                while ((eb * (eb - 1)) / 2 > pidx) --eb;
                while (((eb + 1) * eb) / 2 <= pidx) ++eb;
                int da = pidx - (eb * (eb - 1)) / 2;
                int d = mid + da;
                int e = mid + eb;

                uint64_t zde0 = s_colmask[d][0] ^ s_colmask[e][0];
                uint64_t zde1 = s_colmask[d][1] ^ s_colmask[e][1];
                uint64_t zde2 = s_colmask[d][2] ^ s_colmask[e][2];
                uint16_t cd = s_npi[d], ce = s_npi[e];

                for (int f = e + 1; f < npiv; ++f) {
                    uint64_t r0 = zde0 ^ s_colmask[f][0];
                    uint32_t h = (uint32_t)(r0 & S6_MASK);
                    if ((s_bmap[h >> 5] & (1u << (h & 31u))) == 0u) continue;
                    uint32_t cnt = bkt_cnt[h];
                    if (cnt > S6_SLOTS) cnt = S6_SLOTS;
                    uint64_t r1 = zde1 ^ s_colmask[f][1];
                    uint64_t r2 = zde2 ^ s_colmask[f][2];
                    const stern6_entry_t *baseE = &bkt_entries[(size_t)h * S6_SLOTS];
                    for (uint32_t k = 0; k < cnt; ++k) {
                        uint64_t s0 = baseE[k].s0 ^ r0;
                        int wt = __popcll(s0);
                        if (wt <= inside) {
                            uint64_t s1 = baseE[k].s1 ^ r1;
                            wt += __popcll(s1);
                            if (wt <= inside) {
                                uint64_t s2 = baseE[k].s2 ^ r2;
                                if (wt + __popcll(s2) == inside) {
                                    uint32_t c12 = baseE[k].c12;
                                    uint16_t fc[6] = {
                                        (uint16_t)(c12 & 0xFFFFu), (uint16_t)(c12 >> 16),
                                        (uint16_t)baseE[k].c3, cd, ce, s_npi[f]
                                    };
                                    emit_word_gpu(fc, 6, s0, s1, s2, s_pivots, rows, 3, weight, d_out, d_out_cnt);
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

/* Mode 3 GPU Kernel: Lee-Brickell p=3 (pure shared memory, ultra-fast for small/medium codes) */
__global__ void kernel_lb3(
    int rows, int cols, int nw, int sw_used, int weight,
    int iters_per_block, uint64_t launch_seed,
    found_word_t *d_out, int *d_out_cnt
) {
    __shared__ uint64_t s_sysrow[MAXR][MAXNW];
    __shared__ uint64_t s_colmask[320][SW];
    __shared__ uint16_t s_pivots[MAXR];
    __shared__ uint16_t s_npi[MAXC];
    __shared__ uint8_t s_taken[MAXC];
    __shared__ int s_flag;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int inside = weight - 3;
    if (inside < 0) return;

    uint64_t rng = launch_seed ^ ((uint64_t)(bid + 1) * 0x9e3779b97f4a7c15ULL);
    splitmix64(&rng);

    for (int it = 0; it < iters_per_block; ++it) {
        int npiv = block_sample_systematic(
            &rng, rows, cols, nw, sw_used,
            s_sysrow, s_pivots, s_npi, s_taken, s_colmask, &s_flag
        );
        int npairs = (npiv * (npiv - 1)) / 2;
        for (int pidx = tid; pidx < npairs; pidx += blockDim.x) {
            int b = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
            while ((b * (b - 1)) / 2 > pidx) --b;
            while (((b + 1) * b) / 2 <= pidx) ++b;
            int a = pidx - (b * (b - 1)) / 2;

            uint64_t zab0 = s_colmask[a][0] ^ s_colmask[b][0];
            uint64_t zab1 = s_colmask[a][1] ^ s_colmask[b][1];
            uint64_t zab2 = s_colmask[a][2] ^ s_colmask[b][2];
            uint16_t ca = s_npi[a], cb = s_npi[b];

            for (int c = b + 1; c < npiv; ++c) {
                uint64_t s0 = zab0 ^ s_colmask[c][0];
                int wt = __popcll(s0);
                if (wt > inside) continue;
                uint64_t s1 = (sw_used >= 2) ? (zab1 ^ s_colmask[c][1]) : 0ULL;
                if (sw_used >= 2) {
                    wt += __popcll(s1);
                    if (wt > inside) continue;
                }
                uint64_t s2 = (sw_used >= 3) ? (zab2 ^ s_colmask[c][2]) : 0ULL;
                if (sw_used >= 3) wt += __popcll(s2);
                if (wt == inside) {
                    uint16_t fc[3] = { ca, cb, s_npi[c] };
                    emit_word_gpu(fc, 3, s0, s1, s2, s_pivots, rows, sw_used, weight, d_out, d_out_cnt);
                }
            }
        }
        __syncthreads();
    }
}

static double elapsed_sec(const struct timespec *t0) {
    struct timespec tnow;
    clock_gettime(CLOCK_MONOTONIC, &tnow);
    return (tnow.tv_sec - t0->tv_sec) + 1e-9 * (tnow.tv_nsec - t0->tv_nsec);
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: minwords_gpu <file> <rows> <cols> <target> <weight> <mode> <seed> [max_seconds] [unused]\n");
        return 2;
    }
    const int rows = atoi(argv[2]);
    const int cols = atoi(argv[3]);
    const int target = atoi(argv[4]);
    const int weight = atoi(argv[5]);
    int mode = atoi(argv[6]);
    uint64_t base_seed = strtoull(argv[7], NULL, 10) | 1ULL;
    const double max_seconds = (argc >= 9) ? atof(argv[8]) : 3600.0;

    if (rows <= 0 || rows > MAXR || cols <= 0 || cols > MAXC || weight > MAX_W) {
        fprintf(stderr, "minwords_gpu: unsupported dims rows=%d cols=%d weight=%d\n", rows, cols, weight);
        return 2;
    }
    const int nw = (cols + 63) / 64;
    const int sw_used = (rows + 63) / 64;

    /* Read hex parity-check matrix (identical format to minwords_mt.c) */
    FILE *fh = fopen(argv[1], "r");
    if (!fh) { perror("open"); return 3; }
    static char line[8192];
    memset(h_base, 0, sizeof(h_base));
    for (int i = 0; i < rows; ++i) {
        if (!fgets(line, sizeof(line), fh)) { fprintf(stderr, "short matrix\n"); return 3; }
        int len = (int)strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        for (int p = 0; p < len; ++p) {
            int d = line[len - 1 - p];
            d = (d >= '0' && d <= '9') ? d - '0' : (d | 32) - 'a' + 10;
            for (int b = 0; b < 4; ++b) {
                if ((d >> b) & 1) {
                    int col = 4 * p + b;
                    if (col < cols) h_base[i][col >> 6] |= 1ULL << (col & 63);
                }
            }
        }
    }
    fclose(fh);

    if (mode == 0) {
        mode = (rows >= 88 && weight >= 21) ? 6 : 3;
    } else if (mode == 5 || mode == 7) {
        mode = 6; /* Map all large-code modes to GPU Stern-Dumer p=3+3 */
    }

    /* Optional checkpoint file for Colab persistence */
    const char *ckpt_path = getenv("MINWORDS_CKPT");
    FILE *ckpt_fp = NULL;
    if (ckpt_path && ckpt_path[0]) {
        FILE *rf = fopen(ckpt_path, "r");
        if (rf) {
            uint16_t tmp[MAX_W];
            while (fgets(line, sizeof(line), rf)) {
                int cnt = 0;
                char *p = line;
                while (*p && cnt < MAX_W) {
                    while (*p == ' ' || *p == '\t') ++p;
                    if (*p == '\n' || *p == '\r' || *p == '\0') break;
                    tmp[cnt++] = (uint16_t)strtol(p, &p, 10);
                }
                if (cnt == weight && verify_codeword_host(tmp, weight, rows, cols)) {
                    if (htab_insert(hash_support(tmp, weight))) {
                        ++h_found_count;
                        for (int i = 0; i < weight; ++i) {
                            fprintf(stdout, "%s%u", i ? " " : "", (unsigned)tmp[i]);
                        }
                        fputc('\n', stdout);
                    }
                }
            }
            fclose(rf);
            fflush(stdout);
            fprintf(stderr, "minwords_gpu: resumed %d verified words from %s\n", h_found_count, ckpt_path);
        }
        ckpt_fp = fopen(ckpt_path, "a");
    }

    if (h_found_count >= target) {
        if (ckpt_fp) fclose(ckpt_fp);
        return 0;
    }

    /* Query GPU & size grid */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    int num_blocks = prop.multiProcessorCount;
    if (mode == 6) {
        size_t bytes_per_block = (size_t)S6_BKTS * (sizeof(uint32_t) + (size_t)S6_SLOTS * sizeof(stern6_entry_t));
        int max_by_mem = (int)((free_mem * 75 / 100) / bytes_per_block);
        if (num_blocks > max_by_mem) num_blocks = max_by_mem;
        if (num_blocks < 1) num_blocks = 1;
    } else {
        num_blocks = prop.multiProcessorCount * 4;
    }

    fprintf(stderr, "minwords_gpu: GPU=%s (SMs=%d) mode=%d blocks=%d rows=%d cols=%d w=%d target=%d\n",
            prop.name, prop.multiProcessorCount, mode, num_blocks, rows, cols, weight, target);

    CUDA_CHECK(cudaMemcpyToSymbol(c_base, h_base, sizeof(h_base)));

    uint32_t *d_bkt_cnt = NULL;
    stern6_entry_t *d_bkt_entries = NULL;
    if (mode == 6) {
        CUDA_CHECK(cudaMalloc(&d_bkt_cnt, (size_t)num_blocks * S6_BKTS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_bkt_entries, (size_t)num_blocks * S6_BKTS * S6_SLOTS * sizeof(stern6_entry_t)));
    }

    found_word_t *d_out = NULL;
    int *d_out_cnt = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, MAX_OUT_PER_LAUNCH * sizeof(found_word_t)));
    CUDA_CHECK(cudaMalloc(&d_out_cnt, sizeof(int)));

    found_word_t *h_out = (found_word_t *)malloc(MAX_OUT_PER_LAUNCH * sizeof(found_word_t));

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int iters_per_block = (mode == 6) ? 16 : 64;
    uint64_t total_iters = 0;
    uint64_t launch_idx = 0;
    double last_log = 0.0;

    while (h_found_count < target) {
        double dt = elapsed_sec(&t0);
        if (dt > max_seconds) break;

        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_out_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        uint64_t lseed = base_seed ^ ((++launch_idx) * 0x517cc1b727220a95ULL);

        if (mode == 6) {
            kernel_stern6<<<num_blocks, THREADS_PER_BLOCK>>>(
                rows, cols, nw, sw_used, weight, iters_per_block, lseed,
                d_bkt_cnt, d_bkt_entries, d_out, d_out_cnt
            );
        } else {
            kernel_lb3<<<num_blocks, THREADS_PER_BLOCK>>>(
                rows, cols, nw, sw_used, weight, iters_per_block, lseed,
                d_out, d_out_cnt
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        total_iters += (uint64_t)num_blocks * iters_per_block;

        int n_raw = 0;
        CUDA_CHECK(cudaMemcpy(&n_raw, d_out_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        if (n_raw > MAX_OUT_PER_LAUNCH) n_raw = MAX_OUT_PER_LAUNCH;
        if (n_raw > 0) {
            CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n_raw * sizeof(found_word_t), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_raw && h_found_count < target; ++i) {
                if (h_out[i].idx[0] == 0xFFFF) continue;
                if (!verify_codeword_host(h_out[i].idx, weight, rows, cols)) continue;
                uint64_t key = hash_support(h_out[i].idx, weight);
                if (!htab_insert(key)) continue;

                ++h_found_count;
                for (int j = 0; j < weight; ++j) {
                    fprintf(stdout, "%s%u", j ? " " : "", (unsigned)h_out[i].idx[j]);
                    if (ckpt_fp) fprintf(ckpt_fp, "%s%u", j ? " " : "", (unsigned)h_out[i].idx[j]);
                }
                fputc('\n', stdout);
                if (ckpt_fp) fputc('\n', ckpt_fp);
            }
            fflush(stdout);
            if (ckpt_fp) fflush(ckpt_fp);
        }

        double now = elapsed_sec(&t0);
        if (now - last_log >= 5.0 || h_found_count >= target) {
            double rate_it = (now > 0) ? (double)total_iters / now : 0.0;
            double rate_w = (now > 0) ? (double)h_found_count / now : 0.0;
            double eta = (rate_w > 0 && h_found_count < target) ? (target - h_found_count) / rate_w : 0.0;
            fprintf(stderr, "[%.1fs] found=%d/%d | iters=%llu (%.0f it/s) | %.2f words/s | ETA=%.1fm\n",
                    now, h_found_count, target, (unsigned long long)total_iters, rate_it, rate_w, eta / 60.0);
            last_log = now;
        }
    }

    if (ckpt_fp) fclose(ckpt_fp);
    free(h_out);
    if (d_bkt_cnt) cudaFree(d_bkt_cnt);
    if (d_bkt_entries) cudaFree(d_bkt_entries);
    cudaFree(d_out);
    cudaFree(d_out_cnt);

    fprintf(stderr, "minwords_gpu: DONE mode=%d found=%d/%d in %.2fs (total_iters=%llu)\n",
            mode, h_found_count, target, elapsed_sec(&t0), (unsigned long long)total_iters);
    return (h_found_count > 0) ? 0 : 4;
}
