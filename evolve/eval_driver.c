/* eval_driver.c — fitness function for the AlphaEvolve-lite kernel evolve-loop.
 *
 * This is the automatic evaluator h() from AlphaEvolve, applied to the FastLanes
 * transposed decode+filter+count kernel. It runs the AlphaEvolve "evaluation cascade":
 *   tier 0  compile            (done by evaluate.sh before this binary exists)
 *   tier 1  CORRECTNESS oracle  candidate count == brute-force ground truth, all (b,sigma)
 *   tier 2  THROUGHPUT          median values/ns across a bit-width sweep (the fitness)
 * A candidate that fails tier 1 scores -inf (correct:false) and is never timed for ranking.
 *
 * The candidate provides exactly one symbol:
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target);
 * fl_encode (the layout) is reused from the repo's src/codecs.c so we only evolve the SCAN.
 *
 * Output: one JSON line to stdout. No mocks — real column, real encode, real timing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/common.h"
#include "../src/codecs.h"

/* the evolved kernel, linked in from the candidate .c file */
uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target);

/* exact-selectivity column build (non-clustered): round(sigma*n) values == target. */
static void build_column(uint32_t *vals, size_t n, unsigned b, double sigma,
                         uint32_t target, rng_t *rng) {
    uint32_t m = (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        uint32_t v = (uint32_t)rng_next(rng) & m;
        if (v == target) v = (target + 1) & m;
        vals[i] = v;
    }
    size_t want = (size_t)(sigma * (double)n + 0.5);
    for (size_t i = 0; i < want; i++) vals[i] = target;
    for (size_t i = n - 1; i > 0; i--) {            /* Fisher-Yates */
        size_t j = (size_t)(rng_next(rng) % (i + 1));
        uint32_t t = vals[i]; vals[i] = vals[j]; vals[j] = t;
    }
}

int main(int argc, char **argv) {
    size_t N = argc > 1 ? (size_t)atoll(argv[1]) : 16u * 1024 * 1024; /* mult. of 1024 */
    int repeats = argc > 2 ? atoi(argv[2]) : 7;
    double freq = getenv("BBS_FREQ_GHZ") ? atof(getenv("BBS_FREQ_GHZ")) : 4.0;
    N -= (N % FL_BLOCK);

    unsigned widths[32]; int NB = 0;
    const char *wenv = getenv("BBS_WIDTHS");
    if (wenv && *wenv) {                       /* comma-separated override, e.g. "1,2,3,4,5,6,7" */
        char buf[256]; strncpy(buf, wenv, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
        for (char *t = strtok(buf, ","); t && NB < 32; t = strtok(NULL, ",")) {
            int w = atoi(t); if (w >= 1 && w <= 32) widths[NB++] = (unsigned)w;
        }
    }
    if (NB == 0) { unsigned d[] = {3,5,7,8,12,16,24,32};
        for (NB = 0; NB < 8; NB++) widths[NB] = d[NB]; }
    double sigmas[] = {0.5, 0.1, 0.01}; /* oracle sweep; fitness measured at sigma=0.5 */
    const int NS = (int)(sizeof(sigmas) / sizeof(sigmas[0]));

    rng_t rng; rng_seed(&rng, 0x9E3779B97F4A7C15ull);
    uint32_t *vals   = aligned_alloc64(N * 4);
    uint32_t *planar = aligned_alloc64(((N / FL_BLOCK) * 32 * 32 + 64) * 4);
    double *times = malloc(sizeof(double) * repeats);
    if (!vals || !planar || !times) { printf("{\"correct\":false,\"error\":\"alloc\"}\n"); return 2; }

    /* ---- tier 1: correctness oracle across (b, sigma) ---- */
    for (int bi = 0; bi < NB; bi++) {
        unsigned b = widths[bi];
        for (int si = 0; si < NS; si++) {
            build_column(vals, N, b, sigmas[si], 1, &rng);
            fl_encode(planar, vals, N, b);
            uint64_t brute = 0; for (size_t i = 0; i < N; i++) brute += (vals[i] == 1);
            uint64_t got = fl_scan_candidate(planar, N, b, 1);
            if (got != brute) {
                printf("{\"correct\":false,\"error\":\"oracle mismatch b=%u sigma=%.3f got=%llu want=%llu\"}\n",
                       b, sigmas[si], (unsigned long long)got, (unsigned long long)brute);
                return 0;
            }
        }
    }

    /* ---- tier 2: throughput at sigma=0.5 (the fitness basket) ---- */
    printf("{\"correct\":true,\"N\":%zu,\"repeats\":%d,\"freq_ghz\":%.2f,\"per_b\":{", N, repeats, freq);
    double log_sum = 0; int counted = 0;
    for (int bi = 0; bi < NB; bi++) {
        unsigned b = widths[bi];
        build_column(vals, N, b, 0.5, 1, &rng);
        fl_encode(planar, vals, N, b);
        volatile uint64_t sink = fl_scan_candidate(planar, N, b, 1); /* warm */
        (void)sink;
        for (int r = 0; r < repeats; r++) {
            double t0 = now_ns();
            volatile uint64_t c = fl_scan_candidate(planar, N, b, 1);
            double t1 = now_ns();
            (void)c; times[r] = t1 - t0;
        }
        double med = median_d(times, repeats);
        double vpns = (double)N / med;          /* values per nanosecond (freq-independent) */
        double vpc  = vpns / freq;              /* values per cycle (at stated freq) */
        printf("%s\"%u\":{\"vpns\":%.3f,\"vpc\":%.3f}", bi ? "," : "", b, vpns, vpc);
        log_sum += __builtin_log(vpns); counted++;
    }
    double geo_vpns = __builtin_exp(log_sum / counted);
    printf("},\"geomean_vpns\":%.3f,\"geomean_vpc\":%.3f}\n", geo_vpns, geo_vpns / freq);
    return 0;
}
