/* test_codecs.c — correctness oracle for the encodings + scan kernels (Stage-3 layer 2).
 * Real fixtures, no mocks. Exits non-zero on any failure. */
#include "../src/codecs.h"
#include "../src/common.h"
#include <stdio.h>
#include <assert.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

static uint64_t brute_count(const uint32_t *v, size_t n, uint32_t t) {
    uint64_t c = 0; for (size_t i = 0; i < n; i++) c += (v[i] == t); return c;
}

int main(void) {
    rng_t rng; rng_seed(&rng, 42);
    const size_t N = 100000;
    uint32_t *vals = malloc(N * 4);
    uint32_t *dec  = malloc(N * 4);
    uint32_t *packed = malloc(bp_bytes(N, 32) + 64);

    /* 1. bitpack round-trip + scan agreement for every width 1..32 */
    for (unsigned b = 1; b <= 32; b++) {
        uint32_t m = (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
        for (size_t i = 0; i < N; i++) vals[i] = (uint32_t)rng_next(&rng) & m;
        bp_encode(packed, vals, N, b);
        bp_decode(dec, packed, N, b);
        int ok = 1; for (size_t i = 0; i < N; i++) if (dec[i] != vals[i]) { ok = 0; break; }
        CHECK(ok, "bitpack round-trip");

        uint32_t target = vals[7] & m;
        uint64_t bf = brute_count(vals, N, target);
        CHECK(bp_scan_branchy(packed, N, b, target) == bf, "bp_scan_branchy == brute");
        CHECK(bp_scan_branchfree(packed, N, b, target) == bf, "bp_scan_branchfree == brute");
        CHECK(bp_scan_vec_branchy(packed, N, b, target) == bf, "bp_scan_vec_branchy == brute");
        CHECK(bp_scan_vec_branchfree(packed, N, b, target) == bf, "bp_scan_vec_branchfree == brute");
    }
    printf("ok: bitpack widths 1..32 (round-trip + branchy==branchfree==brute)\n");

    /* 2. byte-aligned fast path must equal the generic decode (NC: no fast-path drift) */
    for (unsigned b = 8; b <= 32; b += 8) {
        uint32_t m = (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
        for (size_t i = 0; i < N; i++) vals[i] = (uint32_t)rng_next(&rng) & m;
        bp_encode(packed, vals, N, b);
        uint32_t target = 5 & m;
        uint64_t fast = bp_scan_branchfree(packed, N, b, target);
        /* force generic path by re-deriving via decode+brute */
        bp_decode(dec, packed, N, b);
        CHECK(fast == brute_count(dec, N, target), "byte-aligned fast path == generic");
    }
    printf("ok: byte-aligned fast paths agree with generic\n");

    /* 2b. FastLanes transposed layout: round-trip + scan == brute (N divisible by 1024) */
    size_t NFL = (N / 1024) * 1024;
    for (unsigned b = 1; b <= 32; b++) {
        uint32_t m = (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
        for (size_t i = 0; i < NFL; i++) vals[i] = (uint32_t)rng_next(&rng) & m;
        uint32_t *planar = malloc((NFL / 1024 * (size_t)b * 32 + 64) * 4);
        fl_encode(planar, vals, NFL, b);
        fl_decode(dec, planar, NFL, b);
        int ok = 1; for (size_t i = 0; i < NFL; i++) if (dec[i] != vals[i]) { ok = 0; break; }
        CHECK(ok, "fastlanes round-trip");
        uint32_t target = vals[123] & m;
        uint64_t bf = brute_count(vals, NFL, target);
        CHECK(fl_scan_branchfree(planar, NFL, b, target) == bf, "fl_scan_branchfree == brute");
        CHECK(fl_scan_branchy(planar, NFL, b, target) == bf, "fl_scan_branchy == brute");
        free(planar);
    }
    printf("ok: fastlanes transposed layout widths 1..32 (round-trip + scan == brute)\n");

    /* 3. RLE round-trip + scan */
    {
        unsigned b = 4; uint32_t m = (1u << b) - 1u;
        /* clustered data so runs are meaningful */
        uint32_t cur = 0;
        for (size_t i = 0; i < N; i++) { if (rng_below(&rng, 10) == 0) cur = (uint32_t)rng_next(&rng) & m; vals[i] = cur; }
        rle_run_t *runs = malloc(N * sizeof(rle_run_t));
        size_t k = rle_encode(runs, vals, N);
        /* expand */
        size_t idx = 0; int ok = 1;
        for (size_t r = 0; r < k && ok; r++)
            for (uint32_t j = 0; j < runs[r].runlen; j++) { if (vals[idx++] != runs[r].value) { ok = 0; break; } }
        CHECK(ok && idx == N, "RLE round-trip");
        uint32_t target = 3 & m;
        CHECK(rle_scan(runs, k, target) == brute_count(vals, N, target), "rle_scan == brute");
        CHECK(k < N, "RLE compresses clustered data (#runs < N)");
        free(runs);
    }
    printf("ok: RLE round-trip + scan\n");

    /* 4. zone maps: count matches brute; clustered skips bytes, random does NOT (NC2) */
    {
        unsigned b = 12; uint32_t m = (1u << b) - 1u; size_t block = 1024;
        zone_t *zones = malloc(((N + block - 1) / block) * sizeof(zone_t));
        uint32_t target = 1234 & m;

        /* clustered: each block drawn from a narrow disjoint sub-range -> most blocks skip */
        for (size_t i = 0; i < N; i++) { size_t blk = i / block; vals[i] = (uint32_t)((blk * 37) & m); }
        /* plant the target in exactly one block */
        vals[block * 3 + 5] = target;
        bp_encode(packed, vals, N, b);
        zone_build(zones, vals, N, block);
        size_t br = 0;
        uint64_t zc = zone_scan(packed, zones, N, block, b, target, 1, &br);
        CHECK(zc == brute_count(vals, N, target), "zone_scan count == brute (clustered)");
        CHECK(br < bp_bytes(N, b) / 2, "clustered: zone map skips most bytes");

        /* NC2: random data -> nearly every block spans target -> ~no skipping */
        for (size_t i = 0; i < N; i++) vals[i] = (uint32_t)rng_next(&rng) & m;
        bp_encode(packed, vals, N, b);
        zone_build(zones, vals, N, block);
        size_t br2 = 0;
        zone_scan(packed, zones, N, block, b, target, 1, &br2);
        CHECK(br2 > (size_t)(0.9 * bp_bytes(N, b)), "NC2: random data gives ~no zone skipping");
        free(zones);
    }
    printf("ok: zone maps (clustered skips; NC2 random does not)\n");

    /* 5. materializing select: branchy == predicated == brute (count + positions) */
    {
        unsigned b = 12; uint32_t m = (1u << b) - 1u;
        for (size_t i = 0; i < N; i++) vals[i] = (uint32_t)rng_next(&rng) & m;
        uint32_t target = 9 & m;
        uint32_t *o1 = malloc(N * 4), *o2 = malloc(N * 4);
        uint64_t k1 = sel32_branchy(vals, N, target, o1);
        uint64_t k2 = sel32_predicated(vals, N, target, o2);
        CHECK(k1 == brute_count(vals, N, target), "sel32_branchy count == brute");
        CHECK(k2 == k1, "sel32_predicated count == branchy");
        int ok = (k1 == k2); for (uint64_t i = 0; i < k1 && ok; i++) if (o1[i] != o2[i]) ok = 0;
        CHECK(ok, "sel32 positions identical");
        free(o1); free(o2);
    }
    printf("ok: materializing select (branchy == predicated == brute)\n");

    free(vals); free(dec); free(packed);
    if (failures) { printf("\n%d FAILURE(S)\n", failures); return 1; }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
