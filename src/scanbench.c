/* scanbench.c — roofline micro-benchmark driver for bandwidth-bound columnar scans.
 *
 * Measures empirical bandwidth ceiling (beta), then times fused decode-filter-aggregate
 * scans across encoding/bit-width x predicate-style x selectivity, emitting one CSV row
 * per cell. The reported bandwidth fraction f = read_GBps / beta_scan is the metric the
 * L1 law predicts.
 *
 * Usage:
 *   scanbench beta                      # print beta (scan-read + STREAM-triad)
 *   scanbench sweep   [N] [repeats]     # full encoding x predicate x sigma grid (CSV)
 *   scanbench zone    [N] [repeats]     # zone-map-skip sweep over sigma x clustering (CSV)
 */
#include "codecs.h"
#include "roofline.h"
#include "common.h"
#include <stdio.h>
#include <string.h>

#if defined(__aarch64__)
#define ARCH "arm64"
#elif defined(__x86_64__)
#define ARCH "x86_64"
#else
#define ARCH "other"
#endif

static char MACHINE[64] = "unknown";
static double FREQ_GHZ = 0.0; /* nominal; for values/cycle (documented caveat) */

/* build a column of N values, bit-width b, with EXACTLY round(sigma*N) equal to target.
 * clustered=1 lays equal-valued runs per block so RLE/zone maps can exploit locality. */
static uint32_t build_column(uint32_t *vals, size_t n, unsigned b, double sigma,
                             uint32_t target, int clustered, size_t block, rng_t *rng) {
    uint32_t m = (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
    uint32_t dom = (b < 30) ? (1u << b) : 0x40000000u;
    if (target > m) target &= m;
    if (!clustered) {
        for (size_t i = 0; i < n; i++) {
            uint32_t v = (uint32_t)rng_next(rng) & m;
            if (v == target) v = (target + 1) & m; /* keep non-targets != target */
            vals[i] = v;
        }
    } else {
        /* each block is a single random run value (high locality) */
        uint32_t cur = (uint32_t)rng_next(rng) & m;
        for (size_t i = 0; i < n; i++) {
            if (block && (i % block) == 0) { cur = (uint32_t)rng_next(rng) & m; if (cur == target) cur = (target + 1) & m; }
            vals[i] = cur;
        }
    }
    /* set EXACTLY round(sigma*n) targets, then Fisher-Yates shuffle for random positions
     * (avoids coupon-collector collisions that under-count at high sigma). For clustered
     * data we skip the shuffle so run/zone locality is preserved and instead overwrite a
     * contiguous prefix of each block. */
    size_t want = (size_t)(sigma * (double)n + 0.5);
    if (!clustered) {
        for (size_t i = 0; i < want; i++) vals[i] = target;
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = (size_t)(rng_next(rng) % (i + 1));
            uint32_t t = vals[i]; vals[i] = vals[j]; vals[j] = t;
        }
    } else {
        /* spread targets as a per-block prefix so runs survive */
        size_t placed = 0;
        for (size_t s = 0; s < n && placed < want; s += (block ? block : n)) {
            size_t e = s + (block ? block : n); if (e > n) e = n;
            size_t per = (size_t)(sigma * (double)(e - s) + 0.5);
            for (size_t i = s; i < e && i < s + per && placed < want; i++) { vals[i] = target; placed++; }
        }
    }
    (void)dom;
    return target;
}

/* time a thunk R times, return median ns */
typedef uint64_t (*scan_thunk)(void *ctx);
static double time_median(scan_thunk fn, void *ctx, int repeats, uint64_t *out_count) {
    double *times = malloc(sizeof(double) * repeats);
    uint64_t c = 0;
    for (int r = 0; r < repeats; r++) {
        double t0 = now_ns();
        c = fn(ctx);
        double t1 = now_ns();
        times[r] = t1 - t0;
    }
    double med = median_d(times, repeats);
    free(times);
    if (out_count) *out_count = c;
    return med;
}

/* ---- scan contexts ---- */
typedef struct { const uint32_t *packed; size_t n; unsigned b; uint32_t target; } bp_ctx;
static uint64_t thunk_bp_branchy(void *c){ bp_ctx*x=c; return bp_scan_branchy(x->packed,x->n,x->b,x->target);}
static uint64_t thunk_bp_branchfree(void *c){ bp_ctx*x=c; return bp_scan_branchfree(x->packed,x->n,x->b,x->target);}
static uint64_t thunk_fl_branchy(void *c){ bp_ctx*x=c; return fl_scan_branchy(x->packed,x->n,x->b,x->target);}
static uint64_t thunk_fl_branchfree(void *c){ bp_ctx*x=c; return fl_scan_branchfree(x->packed,x->n,x->b,x->target);}
typedef struct { const rle_run_t *runs; size_t nruns; uint32_t target; } rle_ctx;
static uint64_t thunk_rle(void *c){ rle_ctx*x=c; return rle_scan(x->runs,x->nruns,x->target);}

static double beta_scan_g = 0, beta_stream_g = 0;

static void emit_header(void) {
    printf("machine,arch,encoding,b,predicate,sigma,clustered,n,bytes_read,"
           "time_ns,read_gbps,values_per_ns,values_per_cycle,beta_scan,beta_stream,f,count,expected\n");
}
static void emit_row(const char *enc, unsigned b, const char *pred, double sigma, int clustered,
                     size_t n, size_t bytes_read, double t_ns, uint64_t count, uint64_t expected) {
    double read_gbps = (double)bytes_read / t_ns;          /* bytes/ns = GB/s */
    double vpns = (double)n / t_ns;
    double vpc = FREQ_GHZ > 0 ? vpns / FREQ_GHZ : 0.0;
    double f = beta_scan_g > 0 ? read_gbps / beta_scan_g : 0.0;
    printf("%s,%s,%s,%u,%s,%.5f,%d,%zu,%zu,%.1f,%.3f,%.3f,%.2f,%.2f,%.2f,%.4f,%llu,%llu\n",
           MACHINE, ARCH, enc, b, pred, sigma, clustered, n, bytes_read, t_ns,
           read_gbps, vpns, vpc, beta_scan_g, beta_stream_g, f,
           (unsigned long long)count, (unsigned long long)expected);
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "beta";
    size_t N = argc > 2 ? (size_t)atoll(argv[2]) : 64 * 1000 * 1000; /* 64M values default */
    int repeats = argc > 3 ? atoi(argv[3]) : 7;
    if (getenv("BBS_MACHINE")) strncpy(MACHINE, getenv("BBS_MACHINE"), sizeof(MACHINE) - 1);
    if (getenv("BBS_FREQ_GHZ")) FREQ_GHZ = atof(getenv("BBS_FREQ_GHZ"));

    /* beta: arrays >> LLC. STREAM uses N doubles; scan-read uses N*8 bytes. */
    size_t beta_elems = 64 * 1000 * 1000; /* 512 MB per array */
    beta_stream_g = stream_triad_gbps(beta_elems, 5);
    beta_scan_g   = scan_read_gbps(beta_elems * 8, 5);

    if (strcmp(cmd, "beta") == 0) {
        printf("machine=%s arch=%s\n", MACHINE, ARCH);
        printf("beta_scan_read  = %.2f GB/s\n", beta_scan_g);
        printf("beta_stream_triad = %.2f GB/s\n", beta_stream_g);
        return 0;
    }

    rng_t rng; rng_seed(&rng, 12345);
    uint32_t *vals = aligned_alloc64(N * 4);
    uint32_t *packed = aligned_alloc64(bp_bytes(N, 32) + 256);
    uint32_t *planar = aligned_alloc64((N / 1024 * 32 * 32 + 64) * 4); /* >= b*32/block, b<=32 */
    if (!vals || !packed || !planar) { fprintf(stderr, "alloc failed\n"); return 2; }
    uint32_t target = 1;

    if (strcmp(cmd, "sweep") == 0) {
        emit_header();
        unsigned widths[] = {3, 5, 7, 8, 9, 12, 16, 17, 24, 32};
        double sigmas[] = {1.0, 0.5, 0.1, 0.01, 0.001, 0.0001};
        rle_run_t *runs = aligned_alloc64(N * sizeof(rle_run_t));
        for (size_t bi = 0; bi < sizeof(widths)/sizeof(widths[0]); bi++) {
            unsigned b = widths[bi];
            for (size_t si = 0; si < sizeof(sigmas)/sizeof(sigmas[0]); si++) {
                double sigma = sigmas[si];
                build_column(vals, N, b, sigma, target, 0, 0, &rng);
                bp_encode(packed, vals, N, b);
                uint64_t expected = (size_t)(sigma * (double)N + 0.5);
                size_t bytes = bp_bytes(N, b);
                bp_ctx ctx = { packed, N, b, target };
                uint64_t c1; double t1 = time_median(thunk_bp_branchy, &ctx, repeats, &c1);
                emit_row("bitpack-scalar", b, "branchy", sigma, 0, N, bytes, t1, c1, expected);
                uint64_t c2; double t2 = time_median(thunk_bp_branchfree, &ctx, repeats, &c2);
                emit_row("bitpack-scalar", b, "branchfree", sigma, 0, N, bytes, t2, c2, expected);
                /* FastLanes transposed layout (lane-parallel, SIMD-decodable) */
                fl_encode(planar, vals, N, b);
                bp_ctx fctx = { planar, N, b, target };
                uint64_t c3; double t3 = time_median(thunk_fl_branchy, &fctx, repeats, &c3);
                emit_row("fastlanes", b, "branchy", sigma, 0, N, bytes, t3, c3, expected);
                uint64_t c4; double t4 = time_median(thunk_fl_branchfree, &fctx, repeats, &c4);
                emit_row("fastlanes", b, "branchfree", sigma, 0, N, bytes, t4, c4, expected);
            }
        }
        /* RLE on clustered data across sigma (run structure depends on clustering) */
        for (size_t si = 0; si < sizeof(sigmas)/sizeof(sigmas[0]); si++) {
            double sigma = sigmas[si];
            build_column(vals, N, 12, sigma, target, 1, 4096, &rng);
            size_t k = rle_encode(runs, vals, N);
            uint64_t expected = 0; for (size_t i = 0; i < N; i++) expected += (vals[i] == target);
            rle_ctx ctx = { runs, k, target };
            uint64_t c; double t = time_median(thunk_rle, &ctx, repeats, &c);
            emit_row("rle", 12, "runscan", sigma, 1, N, rle_bytes(k), t, c, expected);
        }
        free(runs);
        return 0;
    }

    if (strcmp(cmd, "zone") == 0) {
        emit_header();
        unsigned b = 16; size_t block = 8192;
        double sigmas[] = {0.5, 0.1, 0.01, 0.001, 0.0001};
        zone_t *zones = aligned_alloc64(((N + block - 1)/block) * sizeof(zone_t));
        for (int clustered = 0; clustered <= 1; clustered++) {
            for (size_t si = 0; si < sizeof(sigmas)/sizeof(sigmas[0]); si++) {
                double sigma = sigmas[si];
                if (clustered) {
                    /* SORTED column: value rises monotonically so each value localizes to a few
                     * blocks -> zone maps can skip. A range predicate [target, target+W) of width
                     * W selects sigma*N rows occupying a contiguous sigma-fraction of blocks. */
                    uint32_t dom = (b < 30) ? (1u << b) : 0x40000000u;
                    for (size_t i = 0; i < N; i++)
                        vals[i] = (uint32_t)(((uint64_t)i * dom) / N);
                    /* pick an eq-target at the sigma-th quantile's block; eq selectivity ~ N/dom */
                    target = (uint32_t)(0.5 * dom);
                } else {
                    build_column(vals, N, b, sigma, target, 0, block, &rng);
                }
                bp_encode(packed, vals, N, b);
                zone_build(zones, vals, N, block);
                uint64_t expected = 0; for (size_t i = 0; i < N; i++) expected += (vals[i] == target);
                /* full scan baseline */
                bp_ctx fc = { packed, N, b, target };
                uint64_t cf; double tf = time_median(thunk_bp_branchfree, &fc, repeats, &cf);
                emit_row("fullscan", b, "branchfree", sigma, clustered, N, bp_bytes(N,b), tf, cf, expected);
                /* zone-skip scan: time it (count bytes actually read) */
                double *times = malloc(sizeof(double)*repeats); size_t br = 0; uint64_t cz = 0;
                for (int r = 0; r < repeats; r++){ double t0=now_ns(); cz=zone_scan(packed,zones,N,block,b,target,1,&br); double t1=now_ns(); times[r]=t1-t0; }
                double tz = median_d(times, repeats); free(times);
                emit_row("zonemap", b, "branchfree", sigma, clustered, N, br, tz, cz, expected);
            }
        }
        free(zones);
        return 0;
    }

    if (strcmp(cmd, "select") == 0) {
        /* H2: branch misprediction vs selectivity, on a decode-free b=32 column. */
        emit_header();
        unsigned b = 32;
        double sigmas[] = {0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001};
        uint32_t *out = aligned_alloc64(N * 4);
        for (size_t si = 0; si < sizeof(sigmas)/sizeof(sigmas[0]); si++) {
            double sigma = sigmas[si];
            build_column(vals, N, b, sigma, target, 0, 0, &rng);
            uint64_t expected = (size_t)(sigma * (double)N + 0.5);
            size_t bytes = (size_t)N * 4; /* reads the full uint32 column */
            double *tb = malloc(sizeof(double)*repeats); uint64_t kb = 0;
            for (int r = 0; r < repeats; r++){ double t0=now_ns(); kb=sel32_branchy(vals,N,target,out); double t1=now_ns(); tb[r]=t1-t0; }
            emit_row("select", b, "branchy", sigma, 0, N, bytes, median_d(tb,repeats), kb, expected); free(tb);
            double *tp = malloc(sizeof(double)*repeats); uint64_t kp = 0;
            for (int r = 0; r < repeats; r++){ double t0=now_ns(); kp=sel32_predicated(vals,N,target,out); double t1=now_ns(); tp[r]=t1-t0; }
            emit_row("select", b, "predicated", sigma, 0, N, bytes, median_d(tp,repeats), kp, expected); free(tp);
        }
        free(out);
        return 0;
    }

    fprintf(stderr, "unknown command: %s\n", cmd);
    return 2;
}
