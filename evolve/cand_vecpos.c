/* cand_vecpos.c — AVX2 vectorized-position FastLanes scan.
 *
 * Key insight: the 32 lanes of a given word are CONTIGUOUS in memory.
 * For output position p, word wp=(p*b)>>5: base[wp*32 + 0..31] are 32 consecutive
 * uint32 = exactly four aligned 256-bit loads. Likewise base[(wp+1)*32 + l].
 * So each position is a pure streaming load of 4x__m256i (low) + 4x__m256i (high),
 * funnel-shift by `off`, mask, compare-eq vs broadcast target, subtract the
 * match-masks into per-lane accumulators. No gather, no scalar.
 *
 * To make off (and thus the AVX2 shift counts) COMPILE-TIME immediates, we
 * specialize the kernel per bit-width b: fl_scan_bN() has b as a template
 * constant, so every (p*b)&31 folds to a constant and srli/slli get immediates.
 * A switch on b dispatches to the right specialization.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define FL_LANES 32
#define FL_BLOCK 1024

/* One position p with compile-time constant b -> compile-time off/wp/immediates. */
#define POS(p)                                                                       \
    do {                                                                            \
        const unsigned bit = (p) * B, wp = bit >> 5, off = bit & 31;                \
        const uint32_t *pw = base + (size_t)wp * FL_LANES;                          \
        if (off == 0) {                                                             \
            for (int q = 0; q < 4; q++) {                                           \
                __m256i lo = _mm256_load_si256((const __m256i *)pw + q);            \
                __m256i v  = _mm256_and_si256(lo, vmask);                           \
                acc[q] = _mm256_sub_epi32(acc[q], _mm256_cmpeq_epi32(v, vtarget));  \
            }                                                                       \
        } else {                                                                    \
            const uint32_t *pw1 = pw + FL_LANES;                                    \
            for (int q = 0; q < 4; q++) {                                           \
                __m256i lo = _mm256_load_si256((const __m256i *)pw + q);            \
                __m256i hi = _mm256_load_si256((const __m256i *)pw1 + q);           \
                __m256i v  = _mm256_or_si256(_mm256_srli_epi32(lo, off),            \
                                             _mm256_slli_epi32(hi, 32 - off));      \
                v = _mm256_and_si256(v, vmask);                                     \
                acc[q] = _mm256_sub_epi32(acc[q], _mm256_cmpeq_epi32(v, vtarget));  \
            }                                                                       \
        }                                                                           \
    } while (0)

#define DEFINE_SCAN(B)                                                              \
    static uint64_t fl_scan_b##B(const uint32_t *planar, size_t blocks,            \
                                 __m256i vmask, __m256i vtarget) {                 \
        uint64_t count = 0;                                                        \
        for (size_t blk = 0; blk < blocks; blk++) {                               \
            const uint32_t *base = planar + blk * (size_t)(B) * FL_LANES;          \
            __m256i acc[4];                                                        \
            acc[0] = _mm256_setzero_si256(); acc[1] = _mm256_setzero_si256();      \
            acc[2] = _mm256_setzero_si256(); acc[3] = _mm256_setzero_si256();      \
            POS(0);  POS(1);  POS(2);  POS(3);  POS(4);  POS(5);  POS(6);  POS(7); \
            POS(8);  POS(9);  POS(10); POS(11); POS(12); POS(13); POS(14); POS(15);\
            POS(16); POS(17); POS(18); POS(19); POS(20); POS(21); POS(22); POS(23);\
            POS(24); POS(25); POS(26); POS(27); POS(28); POS(29); POS(30); POS(31);\
            __m256i s = _mm256_add_epi32(_mm256_add_epi32(acc[0], acc[1]),         \
                                         _mm256_add_epi32(acc[2], acc[3]));        \
            __m128i lo128 = _mm256_castsi256_si128(s);                             \
            __m128i hi128 = _mm256_extracti128_si256(s, 1);                        \
            __m128i s4 = _mm_add_epi32(lo128, hi128);                              \
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1,0,3,2)));   \
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2,3,0,1)));   \
            count += (uint32_t)_mm_cvtsi128_si32(s4);                              \
        }                                                                          \
        return count;                                                              \
    }

/* B is the template constant inside each specialization */
#define B 1
DEFINE_SCAN(1)
#undef B
#define B 2
DEFINE_SCAN(2)
#undef B
#define B 3
DEFINE_SCAN(3)
#undef B
#define B 4
DEFINE_SCAN(4)
#undef B
#define B 5
DEFINE_SCAN(5)
#undef B
#define B 6
DEFINE_SCAN(6)
#undef B
#define B 7
DEFINE_SCAN(7)
#undef B
#define B 8
DEFINE_SCAN(8)
#undef B
#define B 9
DEFINE_SCAN(9)
#undef B
#define B 10
DEFINE_SCAN(10)
#undef B
#define B 11
DEFINE_SCAN(11)
#undef B
#define B 12
DEFINE_SCAN(12)
#undef B
#define B 13
DEFINE_SCAN(13)
#undef B
#define B 14
DEFINE_SCAN(14)
#undef B
#define B 15
DEFINE_SCAN(15)
#undef B
#define B 16
DEFINE_SCAN(16)
#undef B
#define B 17
DEFINE_SCAN(17)
#undef B
#define B 18
DEFINE_SCAN(18)
#undef B
#define B 19
DEFINE_SCAN(19)
#undef B
#define B 20
DEFINE_SCAN(20)
#undef B
#define B 21
DEFINE_SCAN(21)
#undef B
#define B 22
DEFINE_SCAN(22)
#undef B
#define B 23
DEFINE_SCAN(23)
#undef B
#define B 24
DEFINE_SCAN(24)
#undef B
#define B 25
DEFINE_SCAN(25)
#undef B
#define B 26
DEFINE_SCAN(26)
#undef B
#define B 27
DEFINE_SCAN(27)
#undef B
#define B 28
DEFINE_SCAN(28)
#undef B
#define B 29
DEFINE_SCAN(29)
#undef B
#define B 30
DEFINE_SCAN(30)
#undef B
#define B 31
DEFINE_SCAN(31)
#undef B
#define B 32
DEFINE_SCAN(32)
#undef B

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    __m256i vmask   = _mm256_set1_epi32((int)mask_b(b));
    __m256i vtarget = _mm256_set1_epi32((int)target);
    switch (b) {
        case 1:  return fl_scan_b1 (planar, blocks, vmask, vtarget);
        case 2:  return fl_scan_b2 (planar, blocks, vmask, vtarget);
        case 3:  return fl_scan_b3 (planar, blocks, vmask, vtarget);
        case 4:  return fl_scan_b4 (planar, blocks, vmask, vtarget);
        case 5:  return fl_scan_b5 (planar, blocks, vmask, vtarget);
        case 6:  return fl_scan_b6 (planar, blocks, vmask, vtarget);
        case 7:  return fl_scan_b7 (planar, blocks, vmask, vtarget);
        case 8:  return fl_scan_b8 (planar, blocks, vmask, vtarget);
        case 9:  return fl_scan_b9 (planar, blocks, vmask, vtarget);
        case 10: return fl_scan_b10(planar, blocks, vmask, vtarget);
        case 11: return fl_scan_b11(planar, blocks, vmask, vtarget);
        case 12: return fl_scan_b12(planar, blocks, vmask, vtarget);
        case 13: return fl_scan_b13(planar, blocks, vmask, vtarget);
        case 14: return fl_scan_b14(planar, blocks, vmask, vtarget);
        case 15: return fl_scan_b15(planar, blocks, vmask, vtarget);
        case 16: return fl_scan_b16(planar, blocks, vmask, vtarget);
        case 17: return fl_scan_b17(planar, blocks, vmask, vtarget);
        case 18: return fl_scan_b18(planar, blocks, vmask, vtarget);
        case 19: return fl_scan_b19(planar, blocks, vmask, vtarget);
        case 20: return fl_scan_b20(planar, blocks, vmask, vtarget);
        case 21: return fl_scan_b21(planar, blocks, vmask, vtarget);
        case 22: return fl_scan_b22(planar, blocks, vmask, vtarget);
        case 23: return fl_scan_b23(planar, blocks, vmask, vtarget);
        case 24: return fl_scan_b24(planar, blocks, vmask, vtarget);
        case 25: return fl_scan_b25(planar, blocks, vmask, vtarget);
        case 26: return fl_scan_b26(planar, blocks, vmask, vtarget);
        case 27: return fl_scan_b27(planar, blocks, vmask, vtarget);
        case 28: return fl_scan_b28(planar, blocks, vmask, vtarget);
        case 29: return fl_scan_b29(planar, blocks, vmask, vtarget);
        case 30: return fl_scan_b30(planar, blocks, vmask, vtarget);
        case 31: return fl_scan_b31(planar, blocks, vmask, vtarget);
        case 32: return fl_scan_b32(planar, blocks, vmask, vtarget);
        default: return 0;
    }
}
