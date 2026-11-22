// Historical baseline CUDA kernel for Table 1 ablation.
// Derived from MiniMax minwords_mt.c (Copyright (C) 2026 MiniMax, Apache License 2.0,
// https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece).
// See ../../LICENSE and ../../NOTICE for full license and modification details.
//
// GPU Bit-Parallel ISD Sampler v2 (16-byte ulonglong2 Packed Stern-Dumer)
// Drop-in CUDA replacement for MiniMax's minwords_mt (Route A Primal-Shortened ISD)
// Optimized for NVIDIA L4 / A100 / H100.
//
// Key v2 optimizations over v1:
//   - Packs all (ROWS - 17) syndrome bits into a single 16-byte ulonglong2 (2x less HBM traffic,
//     single 128-bit LDG.E.128 / STG.E.128 per bucket entry!) for ROWS <= 145.
//   - Zero index storage in hot bucket table; recovers (a,b,c) only on the ultra-rare exact hit.
//   - 2 blocks per SM (116 blocks on L4, 216 on A100) for full HBM latency hiding.
//   - Exact Host-side parity verification (H * v == 0 and wt(v) == weight) + Drive checkpointing.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 145 /* 145 - 17 = 128 syndrome bits -> fits in 16-byte ulonglong2! */
#define MAXC 512
#define MAXNW (MAXC / 64)
#define SW 3

#define S6_BITS 17
#define S6_BKTS (1u << S6_BITS) /* 131,072 buckets */
#define S6_MASK (S6_BKTS - 1u)
#define S6_SLOTS 4              /* 4 x 16B = 64B = 1 exact cache line per bucket! */

#define THREADS_PER_BLOCK 256
#define MAX_OUT_PER_LAUNCH 4096
#define MAX_W 64

#define HB 22
#define HM ((1u << HB) - 1u)

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

__device__ static inline void emit_word_from_packed(
    uint32_t h, ulonglong2 lp,
    uint64_t r0, uint64_t r1, uint64_t r2,
    uint16_t cd, uint16_t ce, uint16_t cf,
    int mid,
    const uint64_t s_colmask[320][SW],
    const uint16_t *s_npi,
    const uint16_t *s_pivots,
    int rows, int sw_used, int weight,
    found_word_t *d_out, int *d_out_cnt
) {
    /* Unpack left syndrome (ls0, ls1, ls2) from (h, lp.x, lp.y) */
    uint64_t ls0 = (uint64_t)h | (lp.x << 17);
    uint64_t ls1 = (lp.x >> 47) | (lp.y << 17);
    uint64_t ls2 = (lp.y >> 47);

    /* Find (a, b, c) < mid with colmask[a]^colmask[b]^colmask[c] == (ls0, ls1, ls2) */
    uint16_t ca = 0, cb = 0, cc = 0;
    int found = 0;
    for (int a = 0; a < mid && !found; ++a) {
        uint64_t t0 = ls0 ^ s_colmask[a][0];
        uint64_t t1 = ls1 ^ s_colmask[a][1];
        for (int b = a + 1; b < mid && !found; ++b) {
            uint64_t u0 = t0 ^ s_colmask[b][0];
            uint64_t u1 = t1 ^ s_colmask[b][1];
            for (int c = b + 1; c < mid; ++c) {
                if (s_colmask[c][0] == u0 && s_colmask[c][1] == u1) {
                    if (sw_used < 3 || (s_colmask[a][2] ^ s_colmask[b][2] ^ s_colmask[c][2]) == ls2) {
                        ca = s_npi[a]; cb = s_npi[b]; cc = s_npi[c];
                        found = 1;
                        break;
                    }
                }
            }
        }
    }
    if (!found) return;

    int slot = atomicAdd(d_out_cnt, 1);
    if (slot >= MAX_OUT_PER_LAUNCH) return;

    uint16_t idx[MAX_W];
    int n = 0;
    idx[n++] = ca; idx[n++] = cb; idx[n++] = cc;
    idx[n++] = cd; idx[n++] = ce; idx[n++] = cf;

    uint64_t words[3] = { ls0 ^ r0, ls1 ^ r1, ls2 ^ r2 };
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
    for (int i = 1; i < weight; ++i) {
        uint16_t v = idx[i];
        int j = i - 1;
        while (j >= 0 && idx[j] > v) {
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = v;
    }
    for (int i = 0; i < weight; ++i) d_out[slot].idx[i] = idx[i];
}

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
            if (p == rows) { ok = 0; break; }
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
                    for (int q = 0; q < nw; ++q) s_sysrow[tid][q] ^= s_sysrow[rank][q];
                }
            }
            __syncthreads();
        }
        if (ok) break;
    }

    if (tid == 0) {
        int npiv = 0;
        for (int j = 0; j < cols; ++j) {
            if (!s_taken[j]) s_npi[npiv++] = (uint16_t)j;
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

__global__ void kernel_stern6_v2(
    int rows, int cols, int nw, int sw_used, int weight,
    int iters_per_block, uint64_t launch_seed,
    uint32_t * __restrict__ d_bkt_cnt_all,
    ulonglong2 * __restrict__ d_bkt_entries_all,
    found_word_t *d_out, int *d_out_cnt
) {
    __shared__ uint64_t s_sysrow[MAXR][MAXNW];
    __shared__ uint64_t s_colmask[320][SW];
    __shared__ uint16_t s_pivots[MAXR];
    __shared__ uint16_t s_npi[MAXC];
    __shared__ uint8_t s_taken[MAXC];
    __shared__ uint32_t s_bmap[S6_BKTS / 32]; /* 16 KB */
    __shared__ int s_flag;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int inside = weight - 6;
    if (inside < 0) return;

    uint32_t *bkt_cnt = d_bkt_cnt_all + (size_t)bid * S6_BKTS;
    ulonglong2 *bkt_entries = d_bkt_entries_all + (size_t)bid * S6_BKTS * S6_SLOTS;

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

        for (uint32_t i = tid; i < (S6_BKTS / 32u); i += blockDim.x) s_bmap[i] = 0u;
        for (uint32_t i = tid; i < S6_BKTS; i += blockDim.x) bkt_cnt[i] = 0u;
        __syncthreads();

        /* Phase 1: Left triples -> 16-byte packed ulonglong2 */
        int n_lpairs = (mid * (mid - 1)) / 2;
        for (int pidx = tid; pidx < n_lpairs; pidx += blockDim.x) {
            int b = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
            while ((b * (b - 1)) / 2 > pidx) --b;
            while (((b + 1) * b) / 2 <= pidx) ++b;
            int a = pidx - (b * (b - 1)) / 2;

            uint64_t zab0 = s_colmask[a][0] ^ s_colmask[b][0];
            uint64_t zab1 = s_colmask[a][1] ^ s_colmask[b][1];
            uint64_t zab2 = s_colmask[a][2] ^ s_colmask[b][2];

            for (int c = b + 1; c < mid; ++c) {
                uint64_t s0 = zab0 ^ s_colmask[c][0];
                uint32_t h = (uint32_t)(s0 & S6_MASK);
                uint32_t slot = atomicAdd(&bkt_cnt[h], 1u);
                if (slot == 0u) atomicOr(&s_bmap[h >> 5], 1u << (h & 31u));
                if (slot < S6_SLOTS) {
                    uint64_t s1 = zab1 ^ s_colmask[c][1];
                    uint64_t s2 = zab2 ^ s_colmask[c][2];
                    ulonglong2 packed;
                    packed.x = (s0 >> 17) | (s1 << 47);
                    packed.y = (s1 >> 17) | (s2 << 47);
                    bkt_entries[(size_t)h * S6_SLOTS + slot] = packed;
                }
            }
        }
        __syncthreads();

        /* Phase 2: Right triples -> 128-bit coalesced cache-line probe */
        int n_rpairs = (r_len * (r_len - 1)) / 2;
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
                uint64_t rp0 = (r0 >> 17) | (r1 << 47);
                uint64_t rp1 = (r1 >> 17) | (r2 << 47);

                const ulonglong2 *baseE = &bkt_entries[(size_t)h * S6_SLOTS];
                for (uint32_t k = 0; k < cnt; ++k) {
                    ulonglong2 lp = __ldg(&baseE[k]);
                    int wt = __popcll(lp.x ^ rp0) + __popcll(lp.y ^ rp1);
                    if (wt == inside) {
                        emit_word_from_packed(
                            h, lp, r0, r1, r2, cd, ce, s_npi[f],
                            mid, s_colmask, s_npi, s_pivots,
                            rows, sw_used, weight, d_out, d_out_cnt
                        );
                    }
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
        fprintf(stderr, "usage: minwords_gpu_v2 <file> <rows> <cols> <target> <weight> <mode> <seed> [max_seconds] [unused]\n");
        return 2;
    }
    const int rows = atoi(argv[2]);
    const int cols = atoi(argv[3]);
    const int target = atoi(argv[4]);
    const int weight = atoi(argv[5]);
    uint64_t base_seed = strtoull(argv[7], NULL, 10) | 1ULL;
    const double max_seconds = (argc >= 9) ? atof(argv[8]) : 3600.0;

    if (rows <= 0 || rows > MAXR || cols <= 0 || cols > MAXC || weight > MAX_W) {
        fprintf(stderr, "minwords_gpu_v2: unsupported dims rows=%d cols=%d weight=%d (MAXR=%d)\n", rows, cols, weight, MAXR);
        return 2;
    }
    const int nw = (cols + 63) / 64;
    const int sw_used = (rows + 63) / 64;

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
                        for (int i = 0; i < weight; ++i) fprintf(stdout, "%s%u", i ? " " : "", (unsigned)tmp[i]);
                        fputc('\n', stdout);
                    }
                }
            }
            fclose(rf);
            fflush(stdout);
            fprintf(stderr, "minwords_gpu_v2: resumed %d verified words from %s\n", h_found_count, ckpt_path);
        }
        ckpt_fp = fopen(ckpt_path, "a");
    }
    if (h_found_count >= target) { if (ckpt_fp) fclose(ckpt_fp); return 0; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    /* 2 blocks per SM for latency hiding */
    int num_blocks = prop.multiProcessorCount * 2;
    size_t bytes_per_block = (size_t)S6_BKTS * (sizeof(uint32_t) + (size_t)S6_SLOTS * sizeof(ulonglong2));
    int max_by_mem = (int)((free_mem * 75 / 100) / bytes_per_block);
    if (num_blocks > max_by_mem) num_blocks = max_by_mem;
    if (num_blocks < 1) num_blocks = 1;

    fprintf(stderr, "minwords_gpu_v2: GPU=%s (SMs=%d) blocks=%d (%.1f MB HBM) rows=%d cols=%d w=%d target=%d\n",
            prop.name, prop.multiProcessorCount, num_blocks, (num_blocks * bytes_per_block) / 1e6,
            rows, cols, weight, target);

    CUDA_CHECK(cudaMemcpyToSymbol(c_base, h_base, sizeof(h_base)));

    uint32_t *d_bkt_cnt = NULL;
    ulonglong2 *d_bkt_entries = NULL;
    CUDA_CHECK(cudaMalloc(&d_bkt_cnt, (size_t)num_blocks * S6_BKTS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bkt_entries, (size_t)num_blocks * S6_BKTS * S6_SLOTS * sizeof(ulonglong2)));

    found_word_t *d_out = NULL;
    int *d_out_cnt = NULL;
    CUDA_CHECK(cudaMalloc(&d_out, MAX_OUT_PER_LAUNCH * sizeof(found_word_t)));
    CUDA_CHECK(cudaMalloc(&d_out_cnt, sizeof(int)));
    found_word_t *h_out = (found_word_t *)malloc(MAX_OUT_PER_LAUNCH * sizeof(found_word_t));

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int iters_per_block = 16;
    uint64_t total_iters = 0;
    uint64_t launch_idx = 0;
    double last_log = 0.0;

    while (h_found_count < target) {
        if (elapsed_sec(&t0) > max_seconds) break;
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_out_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        uint64_t lseed = base_seed ^ ((++launch_idx) * 0x517cc1b727220a95ULL);

        kernel_stern6_v2<<<num_blocks, THREADS_PER_BLOCK>>>(
            rows, cols, nw, sw_used, weight, iters_per_block, lseed,
            d_bkt_cnt, d_bkt_entries, d_out, d_out_cnt
        );
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
                if (!htab_insert(hash_support(h_out[i].idx, weight))) continue;

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
    cudaFree(d_bkt_cnt);
    cudaFree(d_bkt_entries);
    cudaFree(d_out);
    cudaFree(d_out_cnt);
    fprintf(stderr, "minwords_gpu_v2: DONE found=%d/%d in %.2fs (total_iters=%llu)\n",
            h_found_count, target, elapsed_sec(&t0), (unsigned long long)total_iters);
    return (h_found_count > 0) ? 0 : 4;
}
