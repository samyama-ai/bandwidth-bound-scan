#include "codecs.h"
#include <string.h>

static inline uint32_t mask_b(unsigned b) {
    return (b < 32) ? ((1u << b) - 1u) : 0xFFFFFFFFu;
}

/* ---- bit-packing ---- */
void bp_encode(uint32_t *packed, const uint32_t *values, size_t n, unsigned b) {
    size_t words = bp_words(n, b);
    memset(packed, 0, words * 4);
    if (b == 0) return;
    for (size_t i = 0; i < n; i++) {
        uint64_t bit = (uint64_t)i * b;
        size_t w = (size_t)(bit >> 5);
        unsigned off = (unsigned)(bit & 31);
        uint64_t v = (uint64_t)(values[i] & mask_b(b));
        uint64_t cur = (uint64_t)packed[w] |
                       ((uint64_t)((w + 1 < words) ? packed[w + 1] : 0u) << 32);
        cur |= (v << off);
        packed[w] = (uint32_t)cur;
        if (w + 1 < words) packed[w + 1] = (uint32_t)(cur >> 32);
    }
}

void bp_decode(uint32_t *out, const uint32_t *packed, size_t n, unsigned b) {
    if (b == 0) { memset(out, 0, n * 4); return; }
    size_t words = bp_words(n, b);
    uint32_t m = mask_b(b);
    for (size_t i = 0; i < n; i++) {
        uint64_t bit = (uint64_t)i * b;
        size_t w = (size_t)(bit >> 5);
        unsigned off = (unsigned)(bit & 31);
        uint64_t cur = (uint64_t)packed[w] |
                       ((uint64_t)((w + 1 < words) ? packed[w + 1] : 0u) << 32);
        out[i] = (uint32_t)((cur >> off) & m);
    }
}

/* generic fused decode-and-compare; branchfree=1 selects the always-on form. */
static inline uint64_t bp_scan_generic(const uint32_t *packed, size_t n, unsigned b,
                                       uint32_t target, int branchfree) {
    size_t words = bp_words(n, b);
    uint32_t m = mask_b(b);
    uint64_t count = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bit = (uint64_t)i * b;
        size_t w = (size_t)(bit >> 5);
        unsigned off = (unsigned)(bit & 31);
        uint64_t cur = (uint64_t)packed[w] |
                       ((uint64_t)((w + 1 < words) ? packed[w + 1] : 0u) << 32);
        uint32_t v = (uint32_t)((cur >> off) & m);
        if (branchfree) count += (v == target);
        else if (v == target) count++;
    }
    return count;
}

/* byte-aligned widths coincide with plain uint8/16/32 arrays (LE) -> vectorizable. */
uint64_t bp_scan_branchfree(const uint32_t *packed, size_t n, unsigned b, uint32_t target) {
    if (b == 8) {
        const uint8_t *p = (const uint8_t *)packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) c += (p[i] == (uint8_t)target);
        return c;
    } else if (b == 16) {
        const uint16_t *p = (const uint16_t *)packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) c += (p[i] == (uint16_t)target);
        return c;
    } else if (b == 32) {
        const uint32_t *p = packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) c += (p[i] == target);
        return c;
    }
    return bp_scan_generic(packed, n, b, target, 1);
}

uint64_t bp_scan_branchy(const uint32_t *packed, size_t n, unsigned b, uint32_t target) {
    if (b == 8) {
        const uint8_t *p = (const uint8_t *)packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) if (p[i] == (uint8_t)target) c++;
        return c;
    } else if (b == 16) {
        const uint16_t *p = (const uint16_t *)packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) if (p[i] == (uint16_t)target) c++;
        return c;
    } else if (b == 32) {
        const uint32_t *p = packed; uint64_t c = 0;
        for (size_t i = 0; i < n; i++) if (p[i] == target) c++;
        return c;
    }
    return bp_scan_generic(packed, n, b, target, 0);
}

/* ---- FastLanes-style bit-transposed (lane-parallel) layout ----
 * Block layout (per 1024-value block): b "planes" of 32 uint32, plane[k][l] = the k-th
 * packed word of lane l. Lane l bit-packs its 32 values (indices l, l+32, ... l+992).
 * Block stride in words = b*32. */
void fl_encode(uint32_t *planar, const uint32_t *values, size_t n, unsigned b) {
    size_t blocks = n / FL_BLOCK;
    uint32_t m = mask_b(b);
    memset(planar, 0, (blocks * (size_t)b * FL_LANES + 64) * 4);
    if (b == 0) return;
    for (size_t blk = 0; blk < blocks; blk++) {
        uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        size_t vbase = blk * FL_BLOCK;
        for (unsigned l = 0; l < FL_LANES; l++) {
            for (unsigned j = 0; j < 32; j++) {            /* 32 values per lane */
                uint64_t v = (uint64_t)(values[vbase + l + 32u * j] & m);
                unsigned bit = j * b, wp = bit >> 5, off = bit & 31;
                uint64_t cur = (uint64_t)base[wp * FL_LANES + l] |
                               ((uint64_t)base[(wp + 1) * FL_LANES + l] << 32);
                cur |= (v << off);
                base[wp * FL_LANES + l] = (uint32_t)cur;
                base[(wp + 1) * FL_LANES + l] = (uint32_t)(cur >> 32);
            }
        }
    }
}

void fl_decode(uint32_t *out, const uint32_t *planar, size_t n, unsigned b) {
    size_t blocks = n / FL_BLOCK; uint32_t m = mask_b(b);
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        size_t vbase = blk * FL_BLOCK;
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;
            const uint32_t *pw1 = base + (wp + 1) * FL_LANES;
            for (unsigned l = 0; l < FL_LANES; l++) {
                uint32_t lo = pw[l] >> off;
                uint32_t hi = off ? (pw1[l] << (32 - off)) : 0u;
                out[vbase + l + 32u * p] = (lo | hi) & m;
            }
        }
    }
}

/* the lane loop is a contiguous constant-shift SIMD op (this is the point).
 * The p-loop is fully unrolled so (wp,off) are compile-time constants per position,
 * turning each lane loop into a pure vector shift+mask+compare+reduce. off==0 is a
 * separate constant path (avoids shift-by-32 UB and the spill load). */
uint64_t fl_scan_branchfree(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK; uint32_t m = mask_b(b); uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        uint32_t partial[FL_LANES] = {0};
        #pragma clang loop unroll(full)
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;
            if (off == 0) {
                for (unsigned l = 0; l < FL_LANES; l++)
                    partial[l] += ((pw[l] & m) == target);
            } else {
                const uint32_t *pw1 = base + (wp + 1) * FL_LANES;
                for (unsigned l = 0; l < FL_LANES; l++) {
                    uint32_t v = ((pw[l] >> off) | (pw1[l] << (32 - off))) & m;
                    partial[l] += (v == target);
                }
            }
        }
        for (unsigned l = 0; l < FL_LANES; l++) count += partial[l];
    }
    return count;
}
uint64_t fl_scan_branchy(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK; uint32_t m = mask_b(b); uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        #pragma clang loop unroll(full)
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;
            if (off == 0) {
                for (unsigned l = 0; l < FL_LANES; l++)
                    if ((pw[l] & m) == target) count++;
            } else {
                const uint32_t *pw1 = base + (wp + 1) * FL_LANES;
                for (unsigned l = 0; l < FL_LANES; l++) {
                    uint32_t v = ((pw[l] >> off) | (pw1[l] << (32 - off))) & m;
                    if (v == target) count++;
                }
            }
        }
    }
    return count;
}

/* ---- vectorizable constant-shift decode (FastLanes-style, chunk of 32 = B words) ----
 * For a compile-time width B, 32 consecutive values occupy exactly B 32-bit words, and
 * value j's (word index, shift) are compile-time constants once the j-loop is unrolled,
 * so clang -O3 -march=native emits SIMD. Reads 1 word past the last chunk -> callers
 * allocate >=64B slack. */
#define DEF_BPVEC(B)                                                                      \
static uint64_t bp_vec_bf_##B(const uint32_t *packed, size_t n, uint32_t target) {        \
    const uint32_t mask = (B < 32) ? ((1u << (B)) - 1u) : 0xFFFFFFFFu;                     \
    uint64_t count = 0; size_t chunks = n / 32; const uint32_t *W = packed;                \
    for (size_t c = 0; c < chunks; c++) {                                                 \
        for (int j = 0; j < 32; j++) {                                                    \
            unsigned bit = (unsigned)j * (B); unsigned wi = bit >> 5, off = bit & 31;       \
            uint64_t two = (uint64_t)W[wi] | ((uint64_t)W[wi + 1] << 32);                  \
            uint32_t v = (uint32_t)((two >> off) & mask);                                  \
            count += (v == target);                                                       \
        }                                                                                 \
        W += (B);                                                                         \
    }                                                                                     \
    for (size_t i = chunks * 32; i < n; i++) {                                            \
        uint64_t bit = (uint64_t)i * (B); size_t w = (size_t)(bit >> 5); unsigned o = (unsigned)(bit & 31); \
        uint64_t two = (uint64_t)packed[w] | ((uint64_t)packed[w + 1] << 32);             \
        count += ((uint32_t)((two >> o) & mask) == target);                               \
    }                                                                                     \
    return count;                                                                         \
}                                                                                         \
static uint64_t bp_vec_by_##B(const uint32_t *packed, size_t n, uint32_t target) {        \
    const uint32_t mask = (B < 32) ? ((1u << (B)) - 1u) : 0xFFFFFFFFu;                     \
    uint64_t count = 0; size_t chunks = n / 32; const uint32_t *W = packed;                \
    for (size_t c = 0; c < chunks; c++) {                                                 \
        for (int j = 0; j < 32; j++) {                                                    \
            unsigned bit = (unsigned)j * (B); unsigned wi = bit >> 5, off = bit & 31;       \
            uint64_t two = (uint64_t)W[wi] | ((uint64_t)W[wi + 1] << 32);                  \
            uint32_t v = (uint32_t)((two >> off) & mask);                                  \
            if (v == target) count++;                                                     \
        }                                                                                 \
        W += (B);                                                                         \
    }                                                                                     \
    for (size_t i = chunks * 32; i < n; i++) {                                            \
        uint64_t bit = (uint64_t)i * (B); size_t w = (size_t)(bit >> 5); unsigned o = (unsigned)(bit & 31); \
        uint64_t two = (uint64_t)packed[w] | ((uint64_t)packed[w + 1] << 32);             \
        if ((uint32_t)((two >> o) & mask) == target) count++;                             \
    }                                                                                     \
    return count;                                                                         \
}

DEF_BPVEC(3) DEF_BPVEC(5) DEF_BPVEC(7) DEF_BPVEC(9)
DEF_BPVEC(12) DEF_BPVEC(17) DEF_BPVEC(21) DEF_BPVEC(24)

uint64_t bp_scan_vec_branchfree(const uint32_t *packed, size_t n, unsigned b, uint32_t target) {
    switch (b) {
        case 3: return bp_vec_bf_3(packed, n, target);
        case 5: return bp_vec_bf_5(packed, n, target);
        case 7: return bp_vec_bf_7(packed, n, target);
        case 9: return bp_vec_bf_9(packed, n, target);
        case 12: return bp_vec_bf_12(packed, n, target);
        case 17: return bp_vec_bf_17(packed, n, target);
        case 21: return bp_vec_bf_21(packed, n, target);
        case 24: return bp_vec_bf_24(packed, n, target);
        default: return bp_scan_branchfree(packed, n, b, target); /* 8/16/32 fast paths */
    }
}
uint64_t bp_scan_vec_branchy(const uint32_t *packed, size_t n, unsigned b, uint32_t target) {
    switch (b) {
        case 3: return bp_vec_by_3(packed, n, target);
        case 5: return bp_vec_by_5(packed, n, target);
        case 7: return bp_vec_by_7(packed, n, target);
        case 9: return bp_vec_by_9(packed, n, target);
        case 12: return bp_vec_by_12(packed, n, target);
        case 17: return bp_vec_by_17(packed, n, target);
        case 21: return bp_vec_by_21(packed, n, target);
        case 24: return bp_vec_by_24(packed, n, target);
        default: return bp_scan_branchy(packed, n, b, target);
    }
}

/* ---- materializing select (isolates branch misprediction) ---- */
uint64_t sel32_branchy(const uint32_t *v, size_t n, uint32_t target, uint32_t *out) {
    uint64_t k = 0;
    for (size_t i = 0; i < n; i++) if (v[i] == target) out[k++] = (uint32_t)i;
    return k;
}
uint64_t sel32_predicated(const uint32_t *v, size_t n, uint32_t target, uint32_t *out) {
    uint64_t k = 0;
    for (size_t i = 0; i < n; i++) { out[k] = (uint32_t)i; k += (v[i] == target); }
    return k;
}

/* ---- RLE ---- */
size_t rle_encode(rle_run_t *runs, const uint32_t *values, size_t n) {
    if (n == 0) return 0;
    size_t k = 0;
    uint32_t cur = values[0];
    uint32_t len = 1;
    for (size_t i = 1; i < n; i++) {
        if (values[i] == cur && len < 0xFFFFFFFFu) { len++; }
        else { runs[k].value = cur; runs[k].runlen = len; k++; cur = values[i]; len = 1; }
    }
    runs[k].value = cur; runs[k].runlen = len; k++;
    return k;
}

uint64_t rle_scan(const rle_run_t *runs, size_t nruns, uint32_t target) {
    uint64_t c = 0;
    for (size_t r = 0; r < nruns; r++) c += (runs[r].value == target) ? runs[r].runlen : 0;
    return c;
}

/* ---- zone maps ---- */
void zone_build(zone_t *zones, const uint32_t *values, size_t n, size_t block) {
    size_t nb = (n + block - 1) / block;
    for (size_t z = 0; z < nb; z++) {
        size_t s = z * block, e = s + block; if (e > n) e = n;
        uint32_t mn = 0xFFFFFFFFu, mx = 0;
        for (size_t i = s; i < e; i++) { uint32_t v = values[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        zones[z].mn = mn; zones[z].mx = mx;
    }
}

uint64_t zone_scan(const uint32_t *packed, const zone_t *zones, size_t n, size_t block,
                   unsigned b, uint32_t target, int branchfree, size_t *bytes_read) {
    size_t nb = (n + block - 1) / block;
    uint64_t count = 0;
    size_t bytes = 0;
    uint32_t m = mask_b(b);
    size_t words = bp_words(n, b);
    for (size_t z = 0; z < nb; z++) {
        if (target < zones[z].mn || target > zones[z].mx) continue; /* skip whole block */
        size_t s = z * block, e = s + block; if (e > n) e = n;
        bytes += ((e - s) * (size_t)b + 7) / 8; /* only scanned blocks count */
        for (size_t i = s; i < e; i++) {
            uint64_t bit = (uint64_t)i * b;
            size_t w = (size_t)(bit >> 5); unsigned off = (unsigned)(bit & 31);
            uint64_t cur = (uint64_t)packed[w] |
                           ((uint64_t)((w + 1 < words) ? packed[w + 1] : 0u) << 32);
            uint32_t v = (uint32_t)((cur >> off) & m);
            if (branchfree) count += (v == target); else if (v == target) count++;
        }
    }
    if (bytes_read) *bytes_read = bytes;
    return count;
}
