/* common.h — timing, RNG, aligned alloc. Portable across x86_64 and arm64 (Apple/Linux).
 * Note: build with -std=gnu11 (not c11) so Linux exposes CLOCK_MONOTONIC/posix_memalign. */
#ifndef BBS_COMMON_H
#define BBS_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- wall-clock (monotonic, ns) — the primary clock for GB/s / bandwidth fraction ---- */
static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ---- aligned alloc (64B, cache-line / AVX-512-friendly) ---- */
static inline void *aligned_alloc64(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, ((bytes + 63) / 64) * 64) != 0) return NULL;
    return p;
}

/* ---- xorshift128+ RNG (fast, deterministic, seedable) ---- */
typedef struct { uint64_t s0, s1; } rng_t;
static inline void rng_seed(rng_t *r, uint64_t seed) {
    /* splitmix64 to fill state */
    uint64_t z = seed + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    r->s0 = z ^ (z >> 31);
    z = (seed + 2) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    r->s1 = z ^ (z >> 31);
    if ((r->s0 | r->s1) == 0) r->s1 = 1;
}
static inline uint64_t rng_next(rng_t *r) {
    uint64_t x = r->s0, y = r->s1;
    r->s0 = y;
    x ^= x << 23;
    r->s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return r->s1 + y;
}
/* uniform in [0, n) */
static inline uint32_t rng_below(rng_t *r, uint32_t n) {
    return n ? (uint32_t)(rng_next(r) % n) : 0;
}

/* median of a double array (sorts in place) */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static inline double median_d(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

#endif /* BBS_COMMON_H */
