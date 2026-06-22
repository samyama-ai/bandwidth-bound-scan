/* candidate_gen0.c — SEED kernel (generation 0).
 * Verbatim copy of src/codecs.c:fl_scan_branchfree, renamed fl_scan_candidate.
 * This is the ~2 values/cycle portable-C baseline from paper14. The evolve-loop's
 * job is to produce correct descendants of this that run faster on the same hardware.
 *
 * Contract (must hold for every candidate):
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target)
 *   returns the number of values in the FastLanes-transposed `planar` block-layout
 *   (FL_LANES=32 lanes, FL_BLOCK=1024 values/block, bit-width b) that equal `target`.
 */
#include <stdint.h>
#include <stddef.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
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
