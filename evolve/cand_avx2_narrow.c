/* cand_avx2_narrow.c — AVX2 with narrow (uint16) per-lane accumulators.
 *
 * Layout recap: block at base = planar + blk*b*32. Lane l holds 32 values at
 * positions l, l+32, ..., l+992, bit-packed little-endian. Word w of lane l is at
 * base[w*32 + l]. So for a fixed word index w, base[w*32 .. w*32+31] is a contiguous
 * 32-uint32 vector = one __m256i pair (two 256-bit loads of 8 lanes each ... actually
 * 32 uint32 = 1024 bits = 4 x __m256i). We process all 32 lanes per position with 4
 * 256-bit vectors.
 *
 * Narrow-accumulator trick: each lane sees exactly 32 positions per block, so each
 * lane's per-block match count is <=32 and fits in a uint16 easily (even uint8).
 * _mm256_cmpeq_epi32 yields 0 / 0xFFFFFFFF per 32-bit lane; subtracting that mask
 * accumulates a count of -(-1)=+1 per match. We keep 4 vectors of 32-bit counters
 * (one per quarter of the 32 lanes) but reset them every block, so they never overflow
 * 32 (well within int32). At block end we add the per-lane counters into a 64-bit total.
 *
 * Optimization over seed: the seed does 32x32 = 1024 scalar (load,mask,cmp,add) ops
 * per block with a scalar partial[] array. Here each position is 4 vector ops covering
 * all 32 lanes, and the compare->subtract folds the "==target" and "+=" into the SIMD
 * lanes, cutting the instruction count ~8x and eliminating the scalar reduction loop.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    uint32_t m = mask_b(b);
    const __m256i vmask   = _mm256_set1_epi32((int)m);
    const __m256i vtarget = _mm256_set1_epi32((int)target);
    uint64_t count = 0;

    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;

        /* 4 vectors of 32-bit per-lane counters covering lanes 0-7,8-15,16-23,24-31.
           Reset per block; max +1 per position over 32 positions -> max 32, no overflow.
           Four independent accumulator chains keep the OoO core's ports busy (ILP). */
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;

            __m256i v0, v1, v2, v3;
            if (off == 0) {
                v0 = _mm256_loadu_si256((const __m256i*)(pw + 0));
                v1 = _mm256_loadu_si256((const __m256i*)(pw + 8));
                v2 = _mm256_loadu_si256((const __m256i*)(pw + 16));
                v3 = _mm256_loadu_si256((const __m256i*)(pw + 24));
            } else {
                const uint32_t *pw1 = pw + FL_LANES;
                __m256i a0 = _mm256_loadu_si256((const __m256i*)(pw + 0));
                __m256i a1 = _mm256_loadu_si256((const __m256i*)(pw + 8));
                __m256i a2 = _mm256_loadu_si256((const __m256i*)(pw + 16));
                __m256i a3 = _mm256_loadu_si256((const __m256i*)(pw + 24));
                __m256i b0 = _mm256_loadu_si256((const __m256i*)(pw1 + 0));
                __m256i b1 = _mm256_loadu_si256((const __m256i*)(pw1 + 8));
                __m256i b2 = _mm256_loadu_si256((const __m256i*)(pw1 + 16));
                __m256i b3 = _mm256_loadu_si256((const __m256i*)(pw1 + 24));
                v0 = _mm256_or_si256(_mm256_srli_epi32(a0, off), _mm256_slli_epi32(b0, 32 - off));
                v1 = _mm256_or_si256(_mm256_srli_epi32(a1, off), _mm256_slli_epi32(b1, 32 - off));
                v2 = _mm256_or_si256(_mm256_srli_epi32(a2, off), _mm256_slli_epi32(b2, 32 - off));
                v3 = _mm256_or_si256(_mm256_srli_epi32(a3, off), _mm256_slli_epi32(b3, 32 - off));
            }
            v0 = _mm256_and_si256(v0, vmask);
            v1 = _mm256_and_si256(v1, vmask);
            v2 = _mm256_and_si256(v2, vmask);
            v3 = _mm256_and_si256(v3, vmask);
            /* cmpeq -> 0xFFFFFFFF (=-1) per 32-bit lane on match; subtract adds +1. */
            acc0 = _mm256_sub_epi32(acc0, _mm256_cmpeq_epi32(v0, vtarget));
            acc1 = _mm256_sub_epi32(acc1, _mm256_cmpeq_epi32(v1, vtarget));
            acc2 = _mm256_sub_epi32(acc2, _mm256_cmpeq_epi32(v2, vtarget));
            acc3 = _mm256_sub_epi32(acc3, _mm256_cmpeq_epi32(v3, vtarget));
        }

        /* horizontal sum of all 32 per-lane counters into count */
        __m256i s = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                     _mm256_add_epi32(acc2, acc3));
        __m128i lo = _mm256_castsi256_si128(s);
        __m128i hi = _mm256_extracti128_si256(s, 1);
        __m128i sum = _mm_add_epi32(lo, hi);
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
        count += (uint32_t)_mm_cvtsi128_si32(sum);
    }
    return count;
}
