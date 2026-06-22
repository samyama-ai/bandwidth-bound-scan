/* cand_restructure.c — portable restructuring, no hand intrinsics.
 * Strategy: single block at a time (no extra memory traffic), but break the
 * per-lane accumulator dependency chain by using TWO accumulator banks: even
 * positions add into bankA, odd positions into bankB. The two banks form
 * independent dependency chains so the out-of-order core can run twice the
 * comparison throughput per lane while the AVX2 vectorizer keeps the clean
 * contiguous 32-lane loops. Reduce both banks once per block.
 */
#include <stdint.h>
#include <stddef.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    uint32_t m = mask_b(b);
    uint64_t count = 0;
    size_t stride = (size_t)b * FL_LANES;

    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * stride;
        uint32_t pa[FL_LANES] = {0};
        uint32_t pb[FL_LANES] = {0};
        #pragma clang loop unroll(full)
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;
            uint32_t *acc = (p & 1) ? pb : pa;
            if (off == 0) {
                for (unsigned l = 0; l < FL_LANES; l++)
                    acc[l] += ((pw[l] & m) == target);
            } else {
                const uint32_t *pw1 = base + (wp + 1) * FL_LANES;
                unsigned sh = 32 - off;
                for (unsigned l = 0; l < FL_LANES; l++) {
                    uint32_t v = ((pw[l] >> off) | (pw1[l] << sh)) & m;
                    acc[l] += (v == target);
                }
            }
        }
        for (unsigned l = 0; l < FL_LANES; l++) count += pa[l] + pb[l];
    }
    return count;
}
