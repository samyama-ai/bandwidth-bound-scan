/* cand_avx512_full.c — AVX-512 (512-bit) FastLanes scan kernel.
 *
 * Port of the winning AVX2 gen2 paths to 512-bit <immintrin.h>:
 *   - general widths 1-31 (except 8/16): per-bit-width specialization with
 *     compile-time off/wp/immediates; 32 lanes of a word = 2x __m512i (16 u32
 *     each). Funnel-shift via _mm512_srli/slli_epi32 for off!=0; compare with
 *     _mm512_cmpeq_epi32_mask -> __mmask16 and subtract via maskz_set1 into
 *     multiple independent accumulators for ILP.
 *   - b==8 : _mm512_cmpeq_epi8_mask -> 64-bit mask, popcnt accumulate (64 vals/cmp).
 *   - b==16: _mm512_cmpeq_epi16_mask -> 32-bit mask, popcnt accumulate (32 vals/cmp).
 *   - b==32: multi-bank uint32 streaming cmpeq_epi32_mask (DRAM-bound).
 *
 * Correctness across all widths 3..32 mandatory; the general path covers every
 * width and the byte/halfword/32 paths are exact specializations of it.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* ============================================================================
 * General per-bit-width specialization (widths 1-31 except 8/16).
 * A word spans 32 lanes = 2 x __m512i (16 u32 each). For each output position p
 * with compile-time-constant b, off/wp are compile-time constants, so the
 * shifts become immediate funnel shifts. Two accumulator banks (one per
 * 512-bit half-word) accumulate the per-lane match count (<=32 per block, no
 * overflow in u32 even over many blocks until horizontal sum).
 * ==========================================================================*/

/* One position p, compile-time constant B. Updates acc0/acc1 (one per half). */
#define POS512(p)                                                                    \
    do {                                                                            \
        const unsigned bit = (p) * B, wp = bit >> 5, off = bit & 31;                \
        const uint32_t *pw = base + (size_t)wp * FL_LANES;                          \
        if (off == 0) {                                                             \
            __m512i v0 = _mm512_load_si512((const __m512i *)pw + 0);                \
            __m512i v1 = _mm512_load_si512((const __m512i *)pw + 1);                \
            v0 = _mm512_and_si512(v0, vmask);                                       \
            v1 = _mm512_and_si512(v1, vmask);                                       \
            __mmask16 k0 = _mm512_cmpeq_epi32_mask(v0, vtarget);                    \
            __mmask16 k1 = _mm512_cmpeq_epi32_mask(v1, vtarget);                    \
            acc0 = _mm512_add_epi32(acc0, _mm512_maskz_set1_epi32(k0, 1));          \
            acc1 = _mm512_add_epi32(acc1, _mm512_maskz_set1_epi32(k1, 1));          \
        } else {                                                                    \
            const uint32_t *pw1 = pw + FL_LANES;                                    \
            __m512i lo0 = _mm512_load_si512((const __m512i *)pw + 0);               \
            __m512i lo1 = _mm512_load_si512((const __m512i *)pw + 1);               \
            __m512i hi0 = _mm512_load_si512((const __m512i *)pw1 + 0);              \
            __m512i hi1 = _mm512_load_si512((const __m512i *)pw1 + 1);              \
            __m512i v0 = _mm512_or_si512(_mm512_srli_epi32(lo0, off),               \
                                         _mm512_slli_epi32(hi0, 32 - off));         \
            __m512i v1 = _mm512_or_si512(_mm512_srli_epi32(lo1, off),               \
                                         _mm512_slli_epi32(hi1, 32 - off));         \
            v0 = _mm512_and_si512(v0, vmask);                                       \
            v1 = _mm512_and_si512(v1, vmask);                                       \
            __mmask16 k0 = _mm512_cmpeq_epi32_mask(v0, vtarget);                    \
            __mmask16 k1 = _mm512_cmpeq_epi32_mask(v1, vtarget);                    \
            acc0 = _mm512_add_epi32(acc0, _mm512_maskz_set1_epi32(k0, 1));          \
            acc1 = _mm512_add_epi32(acc1, _mm512_maskz_set1_epi32(k1, 1));          \
        }                                                                           \
    } while (0)

static inline uint64_t hsum512_epu32(__m512i v) {
    return (uint64_t)_mm512_reduce_add_epi32(v);
}

#define DEFINE_SCAN512(B)                                                            \
    static uint64_t fl_scan512_b##B(const uint32_t *planar, size_t blocks,         \
                                    __m512i vmask, __m512i vtarget) {              \
        __m512i tot0 = _mm512_setzero_si512();                                     \
        __m512i tot1 = _mm512_setzero_si512();                                     \
        for (size_t blk = 0; blk < blocks; blk++) {                               \
            const uint32_t *base = planar + blk * (size_t)(B) * FL_LANES;          \
            __m512i acc0 = _mm512_setzero_si512();                                 \
            __m512i acc1 = _mm512_setzero_si512();                                 \
            POS512(0);  POS512(1);  POS512(2);  POS512(3);                         \
            POS512(4);  POS512(5);  POS512(6);  POS512(7);                         \
            POS512(8);  POS512(9);  POS512(10); POS512(11);                        \
            POS512(12); POS512(13); POS512(14); POS512(15);                        \
            POS512(16); POS512(17); POS512(18); POS512(19);                        \
            POS512(20); POS512(21); POS512(22); POS512(23);                        \
            POS512(24); POS512(25); POS512(26); POS512(27);                        \
            POS512(28); POS512(29); POS512(30); POS512(31);                        \
            /* per-block partial counts <=32, accumulate into wide totals; flush  \
             * periodically not needed: u32 lane holds up to ~134M before wrap,   \
             * but counts per lane grow by <=32/block -> safe for any practical N \
             * (2^32/32 = 134M blocks = 137G values). */                          \
            tot0 = _mm512_add_epi32(tot0, acc0);                                   \
            tot1 = _mm512_add_epi32(tot1, acc1);                                   \
        }                                                                          \
        return hsum512_epu32(_mm512_add_epi32(tot0, tot1));                        \
    }

#define B 1
DEFINE_SCAN512(1)
#undef B
#define B 2
DEFINE_SCAN512(2)
#undef B
#define B 3
DEFINE_SCAN512(3)
#undef B
#define B 4
DEFINE_SCAN512(4)
#undef B
#define B 5
DEFINE_SCAN512(5)
#undef B
#define B 6
DEFINE_SCAN512(6)
#undef B
#define B 7
DEFINE_SCAN512(7)
#undef B
#define B 9
DEFINE_SCAN512(9)
#undef B
#define B 10
DEFINE_SCAN512(10)
#undef B
#define B 11
DEFINE_SCAN512(11)
#undef B
#define B 12
DEFINE_SCAN512(12)
#undef B
#define B 13
DEFINE_SCAN512(13)
#undef B
#define B 14
DEFINE_SCAN512(14)
#undef B
#define B 15
DEFINE_SCAN512(15)
#undef B
#define B 17
DEFINE_SCAN512(17)
#undef B
#define B 18
DEFINE_SCAN512(18)
#undef B
#define B 19
DEFINE_SCAN512(19)
#undef B
#define B 20
DEFINE_SCAN512(20)
#undef B
#define B 21
DEFINE_SCAN512(21)
#undef B
#define B 22
DEFINE_SCAN512(22)
#undef B
#define B 23
DEFINE_SCAN512(23)
#undef B
#define B 24
DEFINE_SCAN512(24)
#undef B
#define B 25
DEFINE_SCAN512(25)
#undef B
#define B 26
DEFINE_SCAN512(26)
#undef B
#define B 27
DEFINE_SCAN512(27)
#undef B
#define B 28
DEFINE_SCAN512(28)
#undef B
#define B 29
DEFINE_SCAN512(29)
#undef B
#define B 30
DEFINE_SCAN512(30)
#undef B
#define B 31
DEFINE_SCAN512(31)
#undef B

/* ============================================================================
 * b == 8 : byte-aligned. 8 words/block * 32 lanes = 256 u32 words = 1024 byte
 * values. _mm512_cmpeq_epi8_mask gives a 64-bit mask (64 byte-values/compare);
 * popcnt accumulates. 4 independent compares per iteration for ILP.
 * ==========================================================================*/
static uint64_t scan512_b8(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 8u * 32u;                 /* 256 u32 words/block */
    __m512i tgt = _mm512_set1_epi8((char)(uint8_t)target);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)8 * FL_LANES;
        /* 256 words = 4 x __m512i (16 words each) per 64-word group; iterate. */
        for (uint32_t w = 0; w < words; w += 64) {
            __mmask64 m0 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const __m512i *)(base + w +  0)), tgt);
            __mmask64 m1 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const __m512i *)(base + w + 16)), tgt);
            __mmask64 m2 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const __m512i *)(base + w + 32)), tgt);
            __mmask64 m3 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const __m512i *)(base + w + 48)), tgt);
            count += (uint64_t)__builtin_popcountll(m0) + (uint64_t)__builtin_popcountll(m1)
                   + (uint64_t)__builtin_popcountll(m2) + (uint64_t)__builtin_popcountll(m3);
        }
    }
    return count;
}

/* ============================================================================
 * b == 16 : halfword-aligned. 16 words/block * 32 lanes = 512 u32 = 1024 u16
 * values. _mm512_cmpeq_epi16_mask gives a 32-bit mask (32 halfword-vals/compare);
 * popcnt accumulates.
 * ==========================================================================*/
static uint64_t scan512_b16(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 16u * 32u;                /* 512 u32 words/block */
    __m512i tgt = _mm512_set1_epi16((short)(uint16_t)target);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)16 * FL_LANES;
        for (uint32_t w = 0; w < words; w += 64) {
            __mmask32 m0 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((const __m512i *)(base + w +  0)), tgt);
            __mmask32 m1 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((const __m512i *)(base + w + 16)), tgt);
            __mmask32 m2 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((const __m512i *)(base + w + 32)), tgt);
            __mmask32 m3 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((const __m512i *)(base + w + 48)), tgt);
            count += (uint64_t)__builtin_popcount(m0) + (uint64_t)__builtin_popcount(m1)
                   + (uint64_t)__builtin_popcount(m2) + (uint64_t)__builtin_popcount(m3);
        }
    }
    return count;
}

/* ============================================================================
 * b == 32 : one u32 value per word. 1024 words/block. Multi-bank streaming
 * cmpeq_epi32_mask + popcnt (DRAM-bound). 4 independent banks for ILP.
 * ==========================================================================*/
static uint64_t scan512_b32(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 32u * 32u;                /* 1024 u32 words/block */
    __m512i tgt = _mm512_set1_epi32((int)target);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)32 * FL_LANES;
        for (uint32_t w = 0; w < words; w += 64) {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(_mm512_loadu_si512((const __m512i *)(base + w +  0)), tgt);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(_mm512_loadu_si512((const __m512i *)(base + w + 16)), tgt);
            __mmask16 m2 = _mm512_cmpeq_epi32_mask(_mm512_loadu_si512((const __m512i *)(base + w + 32)), tgt);
            __mmask16 m3 = _mm512_cmpeq_epi32_mask(_mm512_loadu_si512((const __m512i *)(base + w + 48)), tgt);
            count += (uint64_t)__builtin_popcount(m0) + (uint64_t)__builtin_popcount(m1)
                   + (uint64_t)__builtin_popcount(m2) + (uint64_t)__builtin_popcount(m3);
        }
    }
    return count;
}

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    if (b == 0) return 0;
    uint32_t m = mask_b(b);
    __m512i vmask   = _mm512_set1_epi32((int)m);
    __m512i vtarget = _mm512_set1_epi32((int)target);
    switch (b) {
        case 1:  return fl_scan512_b1 (planar, blocks, vmask, vtarget);
        case 2:  return fl_scan512_b2 (planar, blocks, vmask, vtarget);
        case 3:  return fl_scan512_b3 (planar, blocks, vmask, vtarget);
        case 4:  return fl_scan512_b4 (planar, blocks, vmask, vtarget);
        case 5:  return fl_scan512_b5 (planar, blocks, vmask, vtarget);
        case 6:  return fl_scan512_b6 (planar, blocks, vmask, vtarget);
        case 7:  return fl_scan512_b7 (planar, blocks, vmask, vtarget);
        case 8:  return scan512_b8    (planar, blocks, target & m);
        case 9:  return fl_scan512_b9 (planar, blocks, vmask, vtarget);
        case 10: return fl_scan512_b10(planar, blocks, vmask, vtarget);
        case 11: return fl_scan512_b11(planar, blocks, vmask, vtarget);
        case 12: return fl_scan512_b12(planar, blocks, vmask, vtarget);
        case 13: return fl_scan512_b13(planar, blocks, vmask, vtarget);
        case 14: return fl_scan512_b14(planar, blocks, vmask, vtarget);
        case 15: return fl_scan512_b15(planar, blocks, vmask, vtarget);
        case 16: return scan512_b16   (planar, blocks, target & m);
        case 17: return fl_scan512_b17(planar, blocks, vmask, vtarget);
        case 18: return fl_scan512_b18(planar, blocks, vmask, vtarget);
        case 19: return fl_scan512_b19(planar, blocks, vmask, vtarget);
        case 20: return fl_scan512_b20(planar, blocks, vmask, vtarget);
        case 21: return fl_scan512_b21(planar, blocks, vmask, vtarget);
        case 22: return fl_scan512_b22(planar, blocks, vmask, vtarget);
        case 23: return fl_scan512_b23(planar, blocks, vmask, vtarget);
        case 24: return fl_scan512_b24(planar, blocks, vmask, vtarget);
        case 25: return fl_scan512_b25(planar, blocks, vmask, vtarget);
        case 26: return fl_scan512_b26(planar, blocks, vmask, vtarget);
        case 27: return fl_scan512_b27(planar, blocks, vmask, vtarget);
        case 28: return fl_scan512_b28(planar, blocks, vmask, vtarget);
        case 29: return fl_scan512_b29(planar, blocks, vmask, vtarget);
        case 30: return fl_scan512_b30(planar, blocks, vmask, vtarget);
        case 31: return fl_scan512_b31(planar, blocks, vmask, vtarget);
        case 32: return scan512_b32   (planar, blocks, target);
        default: return 0;
    }
}
