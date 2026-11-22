// GPU Bit-Parallel ISD Sampler v4 (L2-Sized Cooperative Stern-Style ISD)
//
// Attribution & License Notice:
//   Derived from MiniMax's CPU OpenMP sampler `minwords_mt.c` (Apache License 2.0,
//   https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece,
//   released Nov 20, 2026). The algorithmic Mode-6 Stern-style design (p = 3 + 3 = 6,
//   l = 17 collision bits, 16 KB occupancy bitmap pre-filter), CLI convention, FNV/Murmur
//   support hashing, and HB = 22 deduplication table structure originate from MiniMax.
//
// Modifications in minwords_gpu_v4.cu (GPU memory-hierarchy engineering):
//   1. 16-byte `ulonglong2` zero-index syndrome packing for mt <= 145 (4 slots/bucket = 64 B payload).
//   2. Cooperative multi-block bucketing over G = 4 concurrent reductions (35.7 MB table+count+bitmap
//      working set, sized to fit nominal NVIDIA L4 50.3 MB / A100 41.9 MB L2 caches).
//   3. Batched full-SM random systematic reduction (`k_sys<<<TOTAL_BATCH, 256>>>`) amortizing
//      Gaussian elimination across all SMs ahead of STEPS_PER_SYNC cooperative build/probe steps.
//   4. Hardened host-side validation (full row rank check, strict input/hex/checkpoint parsing,
//      bounded device retries, non-repeating resume seed mixing, explicit overflow warnings).

#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXR 145
#define MAXC 512
#define MAXNPIV 320
#define MAXNW (MAXC / 64)
#define SW 3

#define S6_BITS 17
#define S6_BKTS (1u << S6_BITS) /* 131,072 buckets */
#define S6_MASK (S6_BKTS - 1u)
#define S6_SLOTS 4              /* 4 x 16 B = 64 B aligned bucket payload */

#define G_GROUPS 4              /* 4 tables + counts + bitmaps = 35,717,120 B (~35.7 MB / 34.1 MiB) */
#define BLOCKS_PER_GROUP 16     /* 64 cooperative blocks total */
#define STEPS_PER_SYNC 64       /* 64 * 4 = 256 reductions per host sync */
#define TOTAL_BATCH (STEPS_PER_SYNC * G_GROUPS)
#define THREADS 256
#define MAX_SYS_RETRIES 1024

#define MAX_OUT_PER_SYNC 4096
#define MAX_W 64
#define HB 22
#define HTAB_SIZE (1u << HB)
#define HM (HTAB_SIZE - 1u)

typedef struct {
    uint16_t idx[MAX_W];
} found_word_t;

typedef struct {
    uint64_t colmask[MAXNPIV][SW];
    uint16_t pivots[MAXR];
    uint16_t npi[MAXNPIV];
    int npiv;
} sys_state_t;

__constant__ uint64_t c_base[MAXR][MAXNW];

static uint64_t h_base[MAXR][MAXNW];
static uint64_t h_htab[HTAB_SIZE];
static uint32_t h_htab_used = 0;
static int h_found_count = 0;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(5);                                                           \
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

static int parse_int_strict(const char *s, int *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_u64_strict(const char *s, uint64_t *out) {
    if (!s || !*s || *s == '-') return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_double_strict(const char *s, double *out) {
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v) || !(v >= 0.0)) return 0;
    *out = v;
    return 1;
}

static int compute_gf2_rank_host(int rows, int cols, int nw) {
    uint64_t tmp[MAXR][MAXNW];
    memcpy(tmp, h_base, sizeof(tmp));
    int r = 0;
    for (int c = 0; c < cols && r < rows; ++c) {
        int w = c >> 6, b = c & 63;
        int piv = -1;
        for (int i = r; i < rows; ++i) {
            if ((tmp[i][w] >> b) & 1ULL) { piv = i; break; }
        }
        if (piv < 0) continue;
        if (piv != r) {
            for (int q = 0; q < nw; ++q) {
                uint64_t t = tmp[r][q];
                tmp[r][q] = tmp[piv][q];
                tmp[piv][q] = t;
            }
        }
        for (int i = 0; i < rows; ++i) {
            if (i != r && ((tmp[i][w] >> b) & 1ULL)) {
                for (int q = 0; q < nw; ++q) tmp[i][q] ^= tmp[r][q];
            }
        }
        ++r;
    }
    return r;
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
    if (h_htab_used >= (HTAB_SIZE * 3u) / 4u) {
        fprintf(stderr, "minwords_gpu_v4: ERROR deduplication table >75%% full (%u entries)\n", h_htab_used);
        exit(5);
    }
    uint32_t slot = (uint32_t)key & HM;
    for (uint32_t probe = 0; probe < HTAB_SIZE; ++probe) {
        uint64_t cur = h_htab[slot];
        if (cur == key) return 0;
        if (cur == 0ULL) {
            h_htab[slot] = key;
            ++h_htab_used;
            return 1;
        }
        slot = (slot + 1u) & HM;
    }
    fprintf(stderr, "minwords_gpu_v4: ERROR deduplication table exhausted\n");
    exit(5);
}

/* Kernel 1: TOTAL_BATCH blocks generate TOTAL_BATCH independent random systematic reductions */
__global__ void k_sys(
    int rows, int cols, int nw, int sw_used, uint64_t step_seed,
    sys_state_t *d_states, int *d_sys_fail
) {
    __shared__ uint64_t s_sysrow[MAXR][MAXNW];
    __shared__ uint16_t s_pivots[MAXR];
    __shared__ uint16_t s_npi[MAXC];
    __shared__ uint8_t s_taken[MAXC];
    __shared__ int s_flag;

    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    uint64_t rng_init = step_seed + (uint64_t)(g + 1) * 0x9e3779b97f4a7c15ULL;
    uint64_t rng = splitmix64(&rng_init);

    int solved = 0;
    for (int attempt = 0; attempt < MAX_SYS_RETRIES; ++attempt) {
        for (int c = tid; c < cols; c += blockDim.x) s_taken[c] = 0;
        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < rows; ++i) {
                int c;
                do { c = (int)(splitmix64(&rng) % (uint64_t)cols); } while (s_taken[c]);
                s_taken[c] = 1;
                s_pivots[i] = (uint16_t)c;
            }
        }
        __syncthreads();
        if (tid < rows) {
            for (int q = 0; q < nw; ++q) s_sysrow[tid][q] = c_base[tid][q];
        }
        __syncthreads();

        int ok = 1;
        for (int rank = 0; rank < rows; ++rank) {
            int col = s_pivots[rank], w = col >> 6, b = col & 63;
            if (tid == 0) {
                int p = rank;
                while (p < rows && (((s_sysrow[p][w] >> b) & 1ULL) == 0ULL)) ++p;
                s_flag = p;
            }
            __syncthreads();
            int p = s_flag;
            if (p == rows) { ok = 0; break; }
            if (p != rank) {
                if (tid < nw) {
                    uint64_t tmp = s_sysrow[p][tid];
                    s_sysrow[p][tid] = s_sysrow[rank][tid];
                    s_sysrow[rank][tid] = tmp;
                }
                __syncthreads();
            }
            if (tid < rows && tid != rank && ((s_sysrow[tid][w] >> b) & 1ULL)) {
                for (int q = 0; q < nw; ++q) s_sysrow[tid][q] ^= s_sysrow[rank][q];
            }
            __syncthreads();
        }
        if (ok) { solved = 1; break; }
    }

    if (!solved) {
        if (tid == 0) {
            d_states[g].npiv = 0;
            atomicAdd(d_sys_fail, 1);
        }
        return;
    }

    if (tid == 0) {
        int npiv = 0;
        for (int j = 0; j < cols; ++j) if (!s_taken[j]) s_npi[npiv++] = (uint16_t)j;
        s_flag = npiv;
        d_states[g].npiv = npiv;
    }
    __syncthreads();

    int npiv = s_flag;
    if (tid < rows) d_states[g].pivots[tid] = s_pivots[tid];
    for (int j = tid; j < npiv; j += blockDim.x) {
        int col = s_npi[j], w = col >> 6, b = col & 63;
        d_states[g].npi[j] = (uint16_t)col;
        uint64_t m0 = 0ULL, m1 = 0ULL, m2 = 0ULL;
        int r0 = (rows < 64) ? rows : 64;
        for (int i = 0; i < r0; ++i) if ((s_sysrow[i][w] >> b) & 1ULL) m0 |= (1ULL << i);
        if (sw_used >= 2) {
            int r1 = (rows < 128) ? rows : 128;
            for (int i = 64; i < r1; ++i) if ((s_sysrow[i][w] >> b) & 1ULL) m1 |= (1ULL << (i - 64));
        }
        if (sw_used >= 3) {
            for (int i = 128; i < rows; ++i) if ((s_sysrow[i][w] >> b) & 1ULL) m2 |= (1ULL << (i - 128));
        }
        d_states[g].colmask[j][0] = m0;
        d_states[g].colmask[j][1] = m1;
        d_states[g].colmask[j][2] = m2;
    }
}

/* Kernel 2: Zero bucket counts and bitmaps */
__global__ void k_zero(uint32_t * __restrict__ d_bkt_cnt, uint32_t * __restrict__ d_bmap) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t total_cnt = G_GROUPS * S6_BKTS;
    for (uint32_t i = idx; i < total_cnt; i += stride) d_bkt_cnt[i] = 0u;
    uint32_t total_bm = G_GROUPS * (S6_BKTS / 32u);
    for (uint32_t i = idx; i < total_bm; i += stride) d_bmap[i] = 0u;
}

/* Kernel 3: Cooperative left-triple build into 4-slot 16-byte packed buckets */
__global__ void k_build(
    const sys_state_t * __restrict__ d_states,
    uint32_t * __restrict__ d_bkt_cnt_all,
    uint32_t * __restrict__ d_bmap_all,
    ulonglong2 * __restrict__ d_bkt_entries_all
) {
    __shared__ uint64_t s_colmask[MAXNPIV / 2][SW];

    const int g = blockIdx.x / BLOCKS_PER_GROUP;
    const int sub_bid = blockIdx.x % BLOCKS_PER_GROUP;
    const int tid = threadIdx.x;
    const int g_tid = sub_bid * blockDim.x + tid;
    const int g_threads = BLOCKS_PER_GROUP * blockDim.x;

    const int npiv = d_states[g].npiv;
    const int mid = npiv / 2;
    if (mid < 3) return;

    for (int j = tid; j < mid; j += blockDim.x) {
        s_colmask[j][0] = d_states[g].colmask[j][0];
        s_colmask[j][1] = d_states[g].colmask[j][1];
        s_colmask[j][2] = d_states[g].colmask[j][2];
    }
    __syncthreads();

    uint32_t *bkt_cnt = d_bkt_cnt_all + (size_t)g * S6_BKTS;
    uint32_t *bmap = d_bmap_all + (size_t)g * (S6_BKTS / 32u);
    ulonglong2 *bkt_entries = d_bkt_entries_all + (size_t)g * S6_BKTS * S6_SLOTS;

    int n_lpairs = (mid * (mid - 1)) / 2;
    for (int pidx = g_tid; pidx < n_lpairs; pidx += g_threads) {
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
            if (slot == 0u) atomicOr(&bmap[h >> 5], 1u << (h & 31u));
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
}

/* Kernel 4: Cooperative right-triple probe with shared 16 KB occupancy bitmap */
__global__ void k_probe(
    int rows, int sw_used, int weight,
    const sys_state_t * __restrict__ d_states,
    const uint32_t * __restrict__ d_bkt_cnt_all,
    const uint32_t * __restrict__ d_bmap_all,
    const ulonglong2 * __restrict__ d_bkt_entries_all,
    found_word_t *d_out, int *d_out_cnt
) {
    __shared__ uint64_t s_colmask[MAXNPIV][SW];
    __shared__ uint32_t s_bmap[S6_BKTS / 32u]; /* 16 KB */

    const int g = blockIdx.x / BLOCKS_PER_GROUP;
    const int sub_bid = blockIdx.x % BLOCKS_PER_GROUP;
    const int tid = threadIdx.x;
    const int g_tid = sub_bid * blockDim.x + tid;
    const int g_threads = BLOCKS_PER_GROUP * blockDim.x;
    const int inside = weight - 6;

    const int npiv = d_states[g].npiv;
    const int mid = npiv / 2;
    const int r_len = npiv - mid;
    if (mid < 3 || r_len < 3) return;

    const uint32_t *bmap_g = d_bmap_all + (size_t)g * (S6_BKTS / 32u);
    for (uint32_t i = tid; i < (S6_BKTS / 32u); i += blockDim.x) s_bmap[i] = bmap_g[i];
    for (int j = tid; j < npiv; j += blockDim.x) {
        s_colmask[j][0] = d_states[g].colmask[j][0];
        s_colmask[j][1] = d_states[g].colmask[j][1];
        s_colmask[j][2] = d_states[g].colmask[j][2];
    }
    __syncthreads();

    const uint32_t *bkt_cnt = d_bkt_cnt_all + (size_t)g * S6_BKTS;
    const ulonglong2 *bkt_entries = d_bkt_entries_all + (size_t)g * S6_BKTS * S6_SLOTS;

    int n_rpairs = (r_len * (r_len - 1)) / 2;
    for (int pidx = g_tid; pidx < n_rpairs; pidx += g_threads) {
        int eb = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pidx)) * 0.5f);
        while ((eb * (eb - 1)) / 2 > pidx) --eb;
        while (((eb + 1) * eb) / 2 <= pidx) ++eb;
        int da = pidx - (eb * (eb - 1)) / 2;
        int d = mid + da;
        int e = mid + eb;

        uint64_t zde0 = s_colmask[d][0] ^ s_colmask[e][0];
        uint64_t zde1 = s_colmask[d][1] ^ s_colmask[e][1];
        uint64_t zde2 = s_colmask[d][2] ^ s_colmask[e][2];

        for (int f = e + 1; f < npiv; ++f) {
            uint64_t r0 = zde0 ^ s_colmask[f][0];
            uint32_t h = (uint32_t)(r0 & S6_MASK);
            if ((s_bmap[h >> 5] & (1u << (h & 31u))) == 0u) continue;
            uint32_t cnt = __ldg(&bkt_cnt[h]);
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
                    /* Reconstruct (a, b, c) from lossless 128-bit + 17-bit syndrome */
                    uint64_t ls0 = (uint64_t)h | (lp.x << 17);
                    uint64_t ls1 = (lp.x >> 47) | (lp.y << 17);
                    uint64_t ls2 = (lp.y >> 47);
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
                                        ca = d_states[g].npi[a];
                                        cb = d_states[g].npi[b];
                                        cc = d_states[g].npi[c];
                                        found = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (found) {
                        int slot = atomicAdd(d_out_cnt, 1);
                        if (slot < MAX_OUT_PER_SYNC) {
                            uint16_t idx[MAX_W];
                            int n = 0;
                            idx[n++] = ca; idx[n++] = cb; idx[n++] = cc;
                            idx[n++] = d_states[g].npi[d];
                            idx[n++] = d_states[g].npi[e];
                            idx[n++] = d_states[g].npi[f];
                            uint64_t words[3] = { ls0 ^ r0, ls1 ^ r1, ls2 ^ r2 };
                            for (int q = 0; q < sw_used; ++q) {
                                uint64_t w = words[q];
                                while (w && n < MAX_W) {
                                    int bit = __ffsll((long long)w) - 1;
                                    int row = (q << 6) + bit;
                                    if (row < rows) idx[n++] = d_states[g].pivots[row];
                                    w &= (w - 1ULL);
                                }
                            }
                            if (n == weight) {
                                for (int i = 1; i < weight; ++i) {
                                    uint16_t v = idx[i];
                                    int j = i - 1;
                                    while (j >= 0 && idx[j] > v) { idx[j + 1] = idx[j]; --j; }
                                    idx[j + 1] = v;
                                }
                                for (int i = 0; i < weight; ++i) d_out[slot].idx[i] = idx[i];
                            } else {
                                d_out[slot].idx[0] = 0xFFFF;
                            }
                        }
                    }
                }
            }
        }
    }
}

static double elapsed_sec(const struct timespec *t0) {
    struct timespec tnow;
    clock_gettime(CLOCK_MONOTONIC, &tnow);
    return (tnow.tv_sec - t0->tv_sec) + 1e-9 * (tnow.tv_nsec - t0->tv_nsec);
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: minwords_gpu_v4 <file> <rows> <cols> <target> <weight> <mode=6> <seed> [max_seconds] [unused]\n");
        return 2;
    }
    int rows = 0, cols = 0, target = 0, weight = 0, mode = 0;
    uint64_t base_seed = 0;
    double max_seconds = 3600.0;

    if (!parse_int_strict(argv[2], &rows) || !parse_int_strict(argv[3], &cols) ||
        !parse_int_strict(argv[4], &target) || !parse_int_strict(argv[5], &weight) ||
        !parse_int_strict(argv[6], &mode) || !parse_u64_strict(argv[7], &base_seed) ||
        (argc >= 9 && !parse_double_strict(argv[8], &max_seconds))) {
        fprintf(stderr, "minwords_gpu_v4: ERROR invalid numeric CLI argument\n");
        return 2;
    }
    if (mode != 6) {
        fprintf(stderr, "minwords_gpu_v4: ERROR only mode=6 (3+3 Stern, l=17) is supported (got %d)\n", mode);
        return 2;
    }
    if (rows <= 0 || rows > MAXR || cols <= rows || cols > MAXC ||
        (cols - rows) < 6 || (cols - rows) > MAXNPIV ||
        target <= 0 || weight < 6 || weight > MAX_W || weight > cols) {
        fprintf(stderr, "minwords_gpu_v4: ERROR unsupported dimensions/parameters "
                        "(rows=%d [1..%d], cols=%d [rows+6..%d], cols-rows=%d [6..%d], target=%d, weight=%d [6..%d])\n",
                rows, MAXR, cols, MAXC, cols - rows, MAXNPIV, target, weight, MAX_W);
        return 2;
    }

    const int nw = (cols + 63) / 64;
    const int sw_used = (rows + 63) / 64;

    FILE *fh = fopen(argv[1], "r");
    if (!fh) { perror("minwords_gpu_v4: open matrix"); return 3; }
    static char line[8192];
    memset(h_base, 0, sizeof(h_base));
    for (int i = 0; i < rows; ++i) {
        if (!fgets(line, sizeof(line), fh)) {
            fprintf(stderr, "minwords_gpu_v4: ERROR unexpected EOF at matrix row %d/%d\n", i, rows);
            fclose(fh);
            return 3;
        }
        int len = (int)strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (len == 0) {
            fprintf(stderr, "minwords_gpu_v4: ERROR empty matrix row %d\n", i);
            fclose(fh);
            return 3;
        }
        for (int p = 0; p < len; ++p) {
            int ch = line[len - 1 - p];
            int d = -1;
            if (ch >= '0' && ch <= '9') d = ch - '0';
            else if (ch >= 'a' && ch <= 'f') d = ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') d = ch - 'A' + 10;
            else {
                fprintf(stderr, "minwords_gpu_v4: ERROR invalid hex character '%c' at row %d\n", ch, i);
                fclose(fh);
                return 3;
            }
            for (int b = 0; b < 4; ++b) {
                if ((d >> b) & 1) {
                    int col = 4 * p + b;
                    if (col >= cols) {
                        fprintf(stderr, "minwords_gpu_v4: ERROR set bit at col %d >= cols (%d) in row %d\n", col, cols, i);
                        fclose(fh);
                        return 3;
                    }
                    h_base[i][col >> 6] |= 1ULL << (col & 63);
                }
            }
        }
    }
    while (fgets(line, sizeof(line), fh)) {
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (*p != '\0') {
            fprintf(stderr, "minwords_gpu_v4: ERROR unexpected trailing data after row %d\n", rows);
            fclose(fh);
            return 3;
        }
    }
    fclose(fh);

    int host_rank = compute_gf2_rank_host(rows, cols, nw);
    if (host_rank != rows) {
        fprintf(stderr, "minwords_gpu_v4: ERROR parity-check matrix is rank-deficient (rank=%d < rows=%d)\n",
                host_rank, rows);
        return 2;
    }

    const char *ckpt_path = getenv("MINWORDS_CKPT");
    FILE *ckpt_fp = NULL;
    int resumed_count = 0;
    uint64_t resumed_step_idx = 0;
    int malformed_ckpt_lines = 0;
    if (ckpt_path && ckpt_path[0]) {
        FILE *rf = fopen(ckpt_path, "r");
        if (rf) {
            uint16_t tmp[MAX_W];
            int last_ch = '\n';
            while (fgets(line, sizeof(line), rf)) {
                size_t l = strlen(line);
                if (l > 0) last_ch = (unsigned char)line[l - 1];
                if (strncmp(line, "# step_idx=", 11) == 0) {
                    uint64_t parsed_step = 0;
                    char *nl = line + 11;
                    size_t sl = strlen(nl);
                    while (sl && (nl[sl - 1] == '\n' || nl[sl - 1] == '\r')) nl[--sl] = 0;
                    if (parse_u64_strict(nl, &parsed_step) && parsed_step > resumed_step_idx) {
                        resumed_step_idx = parsed_step;
                    }
                    continue;
                }
                if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
                int cnt = 0;
                int valid = 1;
                char *p = line;
                while (*p) {
                    while (*p == ' ' || *p == '\t') ++p;
                    if (*p == '\n' || *p == '\r' || *p == '\0') break;
                    if (cnt >= MAX_W) { valid = 0; break; }
                    errno = 0;
                    char *end = NULL;
                    long v = strtol(p, &end, 10);
                    if (errno != 0 || end == p || v < 0 || v >= cols) { valid = 0; break; }
                    tmp[cnt++] = (uint16_t)v;
                    p = end;
                }
                if (valid && cnt == weight && verify_codeword_host(tmp, weight, rows, cols)) {
                    if (htab_insert(hash_support(tmp, weight))) {
                        ++h_found_count;
                        for (int i = 0; i < weight; ++i) fprintf(stdout, "%s%u", i ? " " : "", (unsigned)tmp[i]);
                        fputc('\n', stdout);
                    }
                } else {
                    ++malformed_ckpt_lines;
                }
            }
            if (ferror(rf)) {
                perror("minwords_gpu_v4: read MINWORDS_CKPT");
                fclose(rf);
                return 3;
            }
            fclose(rf);
            fflush(stdout);
            resumed_count = h_found_count;
            fprintf(stderr, "minwords_gpu_v4: resumed %d verified words from %s (step_idx=%llu, ignored_bad_lines=%d)\n",
                    resumed_count, ckpt_path, (unsigned long long)resumed_step_idx, malformed_ckpt_lines);
            ckpt_fp = fopen(ckpt_path, "a");
            if (!ckpt_fp) { perror("minwords_gpu_v4: open MINWORDS_CKPT for append"); return 3; }
            if (last_ch != '\n' && fputc('\n', ckpt_fp) == EOF) {
                perror("minwords_gpu_v4: write newline to MINWORDS_CKPT");
                fclose(ckpt_fp);
                return 3;
            }
        } else {
            ckpt_fp = fopen(ckpt_path, "a");
            if (!ckpt_fp) { perror("minwords_gpu_v4: create MINWORDS_CKPT"); return 3; }
        }
    }
    if (h_found_count >= target) {
        if (ckpt_fp && fclose(ckpt_fp) != 0) { perror("minwords_gpu_v4: close MINWORDS_CKPT"); return 3; }
        return 0;
    }

    uint64_t effective_seed = base_seed;
    if (resumed_count > 0 && resumed_step_idx == 0) {
        /* Fallback offset when resuming legacy checkpoints that lack # step_idx= lines */
        uint64_t mix = base_seed ^ ((uint64_t)resumed_count * 0x9e3779b97f4a7c15ULL);
        effective_seed = splitmix64(&mix);
    }

    int dev = 0;
    const char *dev_env = getenv("MINWORDS_GPU_DEV");
    if (dev_env && dev_env[0] && !parse_int_strict(dev_env, &dev)) {
        fprintf(stderr, "minwords_gpu_v4: ERROR invalid MINWORDS_GPU_DEV='%s'\n", dev_env);
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t ws_bytes = (size_t)G_GROUPS * S6_BKTS * (sizeof(uint32_t) + (size_t)S6_SLOTS * sizeof(ulonglong2))
                    + (size_t)G_GROUPS * (S6_BKTS / 32u) * sizeof(uint32_t);
    fprintf(stderr, "minwords_gpu_v4: GPU=%s (SMs=%d, L2=%.1fMB) G=%d working_set=%.1fMB rows=%d cols=%d w=%d seed=%llu (eff=%llu)\n",
            prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1e6, G_GROUPS, ws_bytes / 1e6,
            rows, cols, weight, (unsigned long long)base_seed, (unsigned long long)effective_seed);
    if ((size_t)prop.l2CacheSize < ws_bytes) {
        fprintf(stderr, "minwords_gpu_v4: WARNING working set (%.1f MB) exceeds nominal GPU L2 cache (%.1f MB)\n",
                ws_bytes / 1e6, prop.l2CacheSize / 1e6);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_base, h_base, sizeof(h_base)));

    sys_state_t *d_states = NULL;
    uint32_t *d_bkt_cnt = NULL;
    uint32_t *d_bmap = NULL;
    ulonglong2 *d_bkt_entries = NULL;
    found_word_t *d_out = NULL;
    int *d_out_cnt = NULL;
    int *d_sys_fail = NULL;

    CUDA_CHECK(cudaMalloc(&d_states, (size_t)TOTAL_BATCH * sizeof(sys_state_t)));
    CUDA_CHECK(cudaMalloc(&d_bkt_cnt, (size_t)G_GROUPS * S6_BKTS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bmap, (size_t)G_GROUPS * (S6_BKTS / 32u) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bkt_entries, (size_t)G_GROUPS * S6_BKTS * S6_SLOTS * sizeof(ulonglong2)));
    CUDA_CHECK(cudaMalloc(&d_out, MAX_OUT_PER_SYNC * sizeof(found_word_t)));
    CUDA_CHECK(cudaMalloc(&d_out_cnt, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sys_fail, sizeof(int)));

    found_word_t *h_out = (found_word_t *)malloc(MAX_OUT_PER_SYNC * sizeof(found_word_t));
    if (!h_out) {
        fprintf(stderr, "minwords_gpu_v4: ERROR host malloc failed\n");
        return 5;
    }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t total_iters = 0;
    uint64_t step_idx = resumed_step_idx;
    uint64_t out_overflows = 0;
    int newly_found = 0;
    double last_log = 0.0;

    while (h_found_count < target) {
        if (elapsed_sec(&t0) > max_seconds) break;
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_out_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sys_fail, &zero, sizeof(int), cudaMemcpyHostToDevice));

        uint64_t bseed_in = effective_seed + (++step_idx) * 0x517cc1b727220a95ULL;
        uint64_t bseed = splitmix64(&bseed_in);
        k_sys<<<TOTAL_BATCH, THREADS>>>(rows, cols, nw, sw_used, bseed, d_states, d_sys_fail);
        CUDA_CHECK(cudaGetLastError());
        for (int s = 0; s < STEPS_PER_SYNC; ++s) {
            sys_state_t *st_ptr = d_states + s * G_GROUPS;
            k_zero<<<G_GROUPS * BLOCKS_PER_GROUP, THREADS>>>(d_bkt_cnt, d_bmap);
            k_build<<<G_GROUPS * BLOCKS_PER_GROUP, THREADS>>>(st_ptr, d_bkt_cnt, d_bmap, d_bkt_entries);
            k_probe<<<G_GROUPS * BLOCKS_PER_GROUP, THREADS>>>(rows, sw_used, weight, st_ptr, d_bkt_cnt, d_bmap, d_bkt_entries, d_out, d_out_cnt);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int n_sys_fail = 0;
        CUDA_CHECK(cudaMemcpy(&n_sys_fail, d_sys_fail, sizeof(int), cudaMemcpyDeviceToHost));
        if (n_sys_fail > 0) {
            fprintf(stderr, "minwords_gpu_v4: WARNING %d reductions hit MAX_SYS_RETRIES=%d in batch %llu\n",
                    n_sys_fail, MAX_SYS_RETRIES, (unsigned long long)step_idx);
        }
        if (n_sys_fail < TOTAL_BATCH) total_iters += (uint64_t)(TOTAL_BATCH - n_sys_fail);

        int n_raw = 0;
        CUDA_CHECK(cudaMemcpy(&n_raw, d_out_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        if (n_raw > MAX_OUT_PER_SYNC) {
            out_overflows += (uint64_t)(n_raw - MAX_OUT_PER_SYNC);
            fprintf(stderr, "minwords_gpu_v4: WARNING d_out overflow (%d > %d); %d candidates dropped\n",
                    n_raw, MAX_OUT_PER_SYNC, n_raw - MAX_OUT_PER_SYNC);
            n_raw = MAX_OUT_PER_SYNC;
        }
        if (n_raw > 0) {
            CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n_raw * sizeof(found_word_t), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_raw && h_found_count < target; ++i) {
                if (h_out[i].idx[0] == 0xFFFF) continue;
                if (!verify_codeword_host(h_out[i].idx, weight, rows, cols)) continue;
                if (!htab_insert(hash_support(h_out[i].idx, weight))) continue;
                ++h_found_count;
                ++newly_found;
                for (int j = 0; j < weight; ++j) {
                    fprintf(stdout, "%s%u", j ? " " : "", (unsigned)h_out[i].idx[j]);
                    if (ckpt_fp && fprintf(ckpt_fp, "%s%u", j ? " " : "", (unsigned)h_out[i].idx[j]) < 0) {
                        perror("minwords_gpu_v4: write MINWORDS_CKPT");
                        return 3;
                    }
                }
                fputc('\n', stdout);
                if (ckpt_fp && fputc('\n', ckpt_fp) == EOF) {
                    perror("minwords_gpu_v4: write MINWORDS_CKPT");
                    return 3;
                }
            }
            fflush(stdout);
            if (ckpt_fp && fflush(ckpt_fp) != 0) {
                perror("minwords_gpu_v4: flush MINWORDS_CKPT");
                return 3;
            }
        }

        double now = elapsed_sec(&t0);
        if (now - last_log >= 5.0 || h_found_count >= target) {
            if (ckpt_fp) {
                if (fprintf(ckpt_fp, "# step_idx=%llu\n", (unsigned long long)step_idx) < 0 || fflush(ckpt_fp) != 0) {
                    perror("minwords_gpu_v4: write step_idx to MINWORDS_CKPT");
                    return 3;
                }
            }
            double rate_it = (now > 0) ? (double)total_iters / now : 0.0;
            double rate_new = (now > 0) ? (double)newly_found / now : 0.0;
            double eta = (rate_new > 0 && h_found_count < target) ? (target - h_found_count) / rate_new : 0.0;
            fprintf(stderr, "[%.1fs] found=%d/%d (+%d new) | iters=%llu (%.0f it/s) | %.2f new_words/s | ETA=%.1fm\n",
                    now, h_found_count, target, newly_found, (unsigned long long)total_iters, rate_it, rate_new, eta / 60.0);
            last_log = now;
        }
    }

    if (ckpt_fp && fclose(ckpt_fp) != 0) {
        perror("minwords_gpu_v4: close MINWORDS_CKPT");
        return 3;
    }
    free(h_out);
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_bkt_cnt));
    CUDA_CHECK(cudaFree(d_bmap));
    CUDA_CHECK(cudaFree(d_bkt_entries));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_out_cnt));
    CUDA_CHECK(cudaFree(d_sys_fail));

    fprintf(stderr, "minwords_gpu_v4: DONE found=%d/%d (+%d new) in %.2fs (total_iters=%llu, out_overflows=%llu)\n",
            h_found_count, target, newly_found, elapsed_sec(&t0),
            (unsigned long long)total_iters, (unsigned long long)out_overflows);
    if (h_found_count >= target) return 0;
    return (h_found_count > 0) ? 1 : 4;
}
