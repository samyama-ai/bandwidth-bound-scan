/* cand_bytealigned.c — byte/halfword/word-aligned SIMD specialization + general fallback.
 *
 * Contract:
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target)
 *
 * Layout recap (from src/codecs.c):
 *   Block blk starts at base = planar + blk*b*32. For output position p (0..31) bit=p*b,
 *   wp=bit>>5, off=bit&31. Word w of lane l is stored at base[w*32 + l] -- so for a fixed
 *   word index w the 32 lanes are contiguous in memory: base[w*32 .. w*32+31].
 *
 * For b in {32,16,8,4,2,1} every output position is word-aligned (off==0). The whole block
 * is then just (b*32) contiguous uint32 words whose b-bit sub-fields we compare against the
 * (already masked) target -- pure streaming AVX2 with no cross-word shifts.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* horizontal sum of 8 x uint32 in a __m256i */
static inline uint64_t hsum_epu32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(s);
}

/* ---- b == 32: every word is one value; compare all words to target. ---- */
static uint64_t scan_b32(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 32u * 32u;                /* 1024 words/block */
    __m256i tgt = _mm256_set1_epi32((int)target);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)32 * FL_LANES;
        __m256i acc = _mm256_setzero_si256();        /* mask sums = -count; 1024 fits in u32 */
        for (uint32_t w = 0; w < words; w += 32) {
            acc = _mm256_sub_epi32(acc, _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)(base + w +  0)), tgt));
            acc = _mm256_sub_epi32(acc, _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)(base + w +  8)), tgt));
            acc = _mm256_sub_epi32(acc, _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)(base + w + 16)), tgt));
            acc = _mm256_sub_epi32(acc, _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i *)(base + w + 24)), tgt));
        }
        count += hsum_epu32(acc);
    }
    return count;
}

/* ---- b == 16: 16 words/block, each word = 2 halfword values. Compare 16-bit lanes. ---- */
static uint64_t scan_b16(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 16u * 32u;                /* 512 words = 1024 values/block */
    __m256i tgt = _mm256_set1_epi16((short)(uint16_t)target);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)16 * FL_LANES;
        /* cmpeq_epi16 yields 0xFFFF lanes; sub into 16-bit accumulators (<=1024 matches: fits) */
        __m256i acc = _mm256_setzero_si256();
        for (uint32_t w = 0; w < words; w += 16) {
            acc = _mm256_sub_epi16(acc, _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)(base + w + 0)), tgt));
            acc = _mm256_sub_epi16(acc, _mm256_cmpeq_epi16(_mm256_loadu_si256((const __m256i *)(base + w + 8)), tgt));
        }
        /* widen 16x u16 -> sum */
        __m256i lo = _mm256_and_si256(acc, _mm256_set1_epi32(0x0000FFFF));
        __m256i hi = _mm256_srli_epi32(acc, 16);
        count += hsum_epu32(_mm256_add_epi32(lo, hi));
    }
    return count;
}

/* ---- b == 8: 8 words/block, each word = 4 byte values. Compare 8-bit lanes. ---- */
static uint64_t scan_b8(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 8u * 32u;                 /* 256 words = 1024 values/block */
    __m256i tgt = _mm256_set1_epi8((char)(uint8_t)target);
    __m256i ones = _mm256_set1_epi8(1);              /* mask & 1 -> 1 per match byte */
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)8 * FL_LANES;
        /* up to 1024 matches/block; sum via SAD on byte-bitmaps to avoid u8 overflow. */
        __m256i acc = _mm256_setzero_si256();        /* per-block: 4 SAD-sums of 64-bit */
        for (uint32_t w = 0; w < words; w += 32) {
            __m256i m0 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(base + w +  0)), tgt), ones);
            __m256i m1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(base + w +  8)), tgt), ones);
            __m256i m2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(base + w + 16)), tgt), ones);
            __m256i m3 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(base + w + 24)), tgt), ones);
            __m256i s01 = _mm256_add_epi8(m0, m1);
            __m256i s23 = _mm256_add_epi8(m2, m3);
            __m256i s = _mm256_add_epi8(s01, s23);   /* max per byte = 4, no overflow */
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(s, _mm256_setzero_si256()));
        }
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi64(lo, hi);
        count += (uint64_t)_mm_cvtsi128_si64(s) + (uint64_t)_mm_extract_epi64(s, 1);
    }
    return count;
}

/* ---- general fallback (off may be nonzero): seed scalar logic with narrow per-lane
 *      counters, which the compiler auto-vectorizes well for small b. ---- */
static uint64_t scan_general(const uint32_t *planar, size_t blocks, unsigned b, uint32_t target) {
    uint32_t m = mask_b(b);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        uint32_t partial[FL_LANES] = {0};  /* seed uses u32; auto-vectorizes best */
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + (size_t)wp * FL_LANES;
            if (off == 0) {
                for (unsigned l = 0; l < FL_LANES; l++)
                    partial[l] += ((pw[l] & m) == target);
            } else {
                const uint32_t *pw1 = base + (size_t)(wp + 1) * FL_LANES;
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

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    if (b == 0) return 0;
    uint32_t m = mask_b(b);
    target &= m;                          /* values are stored masked; normalize target */
    switch (b) {
        case 32: return scan_b32(planar, blocks, target);
        case 16: return scan_b16(planar, blocks, target);
        case 8:  return scan_b8(planar, blocks, target);
        default: return scan_general(planar, blocks, b, target);
    }
}
