/* cand_neon_ilp.c — NEON ILP/MLP-maximizing descendant of the FastLanes scan seed.
 *
 * Contract:
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target)
 *
 * Layout (FastLanes bit-transposed): block blk starts at planar + blk*b*32.
 * Within a block, for output position p (0..31): bit=p*b, wp=bit>>5, off=bit&31.
 * The 32 lanes for a given word w are stored contiguously: base[w*32 + l], l=0..31.
 * So a "word row" = 32 consecutive uint32 = 8 NEON uint32x4 vectors.
 *
 * Strategy: vectorize the 32 lanes (8 uint32x4 vectors per word row) and keep 8
 * independent accumulator chains so the wide M4 NEON backend never stalls on a
 * single dependency chain. We count matches by subtracting the all-ones compare
 * mask (vceqq returns 0xFFFFFFFF on match == -1 as int) so partials += match.
 * Shifts are immediate via per-b specialization (switch on b). Multiple word
 * rows are processed back-to-back with software prefetch of the next block to
 * keep load pipes and memory-level parallelism saturated.
 */
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* horizontal add of 8 uint32x4 accumulator chains into a scalar */
static inline uint64_t reduce8(uint32x4_t a0, uint32x4_t a1, uint32x4_t a2, uint32x4_t a3,
                               uint32x4_t a4, uint32x4_t a5, uint32x4_t a6, uint32x4_t a7) {
    uint32x4_t s0 = vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3));
    uint32x4_t s1 = vaddq_u32(vaddq_u32(a4, a5), vaddq_u32(a6, a7));
    uint32x4_t s  = vaddq_u32(s0, s1);
    return (uint64_t)vaddvq_u32(s);
}

/* General (non-specialized) fallback: scalar, identical semantics to the seed. */
static uint64_t scan_general(const uint32_t *planar, size_t blocks, unsigned b, uint32_t target) {
    uint32_t m = mask_b(b);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        uint32_t partial[FL_LANES] = {0};
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

/* ---- NEON specialized path, parameterized on a compile-time bit width B ---- */
/* Process one block (32 positions), b == B (immediate shifts), 8 accumulator chains.
 * Accumulators are uint32x4; max matches per lane per block is 32 so no overflow. */

#define DECL_ACC \
    uint32x4_t a0=vdupq_n_u32(0),a1=vdupq_n_u32(0),a2=vdupq_n_u32(0),a3=vdupq_n_u32(0); \
    uint32x4_t a4=vdupq_n_u32(0),a5=vdupq_n_u32(0),a6=vdupq_n_u32(0),a7=vdupq_n_u32(0);

/* Aligned, no-boundary-cross path: off == 0. */
#define POS_ALIGNED(WP)                                                                 \
do {                                                                                    \
    const uint32_t *pw = base + (size_t)(WP) * FL_LANES;                                \
    uint32x4_t v0 = vandq_u32(vld1q_u32(pw +  0), mvec);                                \
    uint32x4_t v1 = vandq_u32(vld1q_u32(pw +  4), mvec);                                \
    uint32x4_t v2 = vandq_u32(vld1q_u32(pw +  8), mvec);                                \
    uint32x4_t v3 = vandq_u32(vld1q_u32(pw + 12), mvec);                                \
    uint32x4_t v4 = vandq_u32(vld1q_u32(pw + 16), mvec);                                \
    uint32x4_t v5 = vandq_u32(vld1q_u32(pw + 20), mvec);                                \
    uint32x4_t v6 = vandq_u32(vld1q_u32(pw + 24), mvec);                                \
    uint32x4_t v7 = vandq_u32(vld1q_u32(pw + 28), mvec);                                \
    a0 = vsubq_u32(a0, vceqq_u32(v0, tgt));                                             \
    a1 = vsubq_u32(a1, vceqq_u32(v1, tgt));                                             \
    a2 = vsubq_u32(a2, vceqq_u32(v2, tgt));                                             \
    a3 = vsubq_u32(a3, vceqq_u32(v3, tgt));                                             \
    a4 = vsubq_u32(a4, vceqq_u32(v4, tgt));                                             \
    a5 = vsubq_u32(a5, vceqq_u32(v5, tgt));                                             \
    a6 = vsubq_u32(a6, vceqq_u32(v6, tgt));                                             \
    a7 = vsubq_u32(a7, vceqq_u32(v7, tgt));                                             \
} while (0)

/* Boundary-crossing path: OFF and (32-OFF) must be constant expressions, 1<=OFF<=31. */
#define POS_CROSS(WP, OFF)                                                              \
do {                                                                                    \
    const uint32_t *pw  = base + (size_t)(WP) * FL_LANES;                               \
    const uint32_t *pw1 = pw + FL_LANES;                                                \
    uint32x4_t l0 = vld1q_u32(pw +  0), h0 = vld1q_u32(pw1 +  0);                       \
    uint32x4_t l1 = vld1q_u32(pw +  4), h1 = vld1q_u32(pw1 +  4);                       \
    uint32x4_t l2 = vld1q_u32(pw +  8), h2 = vld1q_u32(pw1 +  8);                       \
    uint32x4_t l3 = vld1q_u32(pw + 12), h3 = vld1q_u32(pw1 + 12);                       \
    uint32x4_t l4 = vld1q_u32(pw + 16), h4 = vld1q_u32(pw1 + 16);                       \
    uint32x4_t l5 = vld1q_u32(pw + 20), h5 = vld1q_u32(pw1 + 20);                       \
    uint32x4_t l6 = vld1q_u32(pw + 24), h6 = vld1q_u32(pw1 + 24);                       \
    uint32x4_t l7 = vld1q_u32(pw + 28), h7 = vld1q_u32(pw1 + 28);                       \
    uint32x4_t v0 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h0, 32-(OFF)), l0, (OFF)), mvec); \
    uint32x4_t v1 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h1, 32-(OFF)), l1, (OFF)), mvec); \
    uint32x4_t v2 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h2, 32-(OFF)), l2, (OFF)), mvec); \
    uint32x4_t v3 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h3, 32-(OFF)), l3, (OFF)), mvec); \
    uint32x4_t v4 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h4, 32-(OFF)), l4, (OFF)), mvec); \
    uint32x4_t v5 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h5, 32-(OFF)), l5, (OFF)), mvec); \
    uint32x4_t v6 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h6, 32-(OFF)), l6, (OFF)), mvec); \
    uint32x4_t v7 = vandq_u32(vsraq_n_u32(vshlq_n_u32(h7, 32-(OFF)), l7, (OFF)), mvec); \
    a0 = vsubq_u32(a0, vceqq_u32(v0, tgt));                                             \
    a1 = vsubq_u32(a1, vceqq_u32(v1, tgt));                                             \
    a2 = vsubq_u32(a2, vceqq_u32(v2, tgt));                                             \
    a3 = vsubq_u32(a3, vceqq_u32(v3, tgt));                                             \
    a4 = vsubq_u32(a4, vceqq_u32(v4, tgt));                                             \
    a5 = vsubq_u32(a5, vceqq_u32(v5, tgt));                                             \
    a6 = vsubq_u32(a6, vceqq_u32(v6, tgt));                                             \
    a7 = vsubq_u32(a7, vceqq_u32(v7, tgt));                                             \
} while (0)

/* Pick the right path at compile time. Both WP and OFF are constant expressions. */
#define POS_BODY(B, p)                                                                  \
do {                                                                                    \
    enum { bit_ = (p) * (B), wp_ = bit_ >> 5, off_ = bit_ & 31 };                       \
    if (off_ == 0) POS_ALIGNED(wp_);                                                    \
    else           POS_CROSS(wp_, (off_ == 0 ? 1 : off_));                              \
} while (0)

/* fully-unrolled 32 positions for a block at width B */
#define BLOCK32(B)         \
    POS_BODY(B,0);  POS_BODY(B,1);  POS_BODY(B,2);  POS_BODY(B,3);  \
    POS_BODY(B,4);  POS_BODY(B,5);  POS_BODY(B,6);  POS_BODY(B,7);  \
    POS_BODY(B,8);  POS_BODY(B,9);  POS_BODY(B,10); POS_BODY(B,11); \
    POS_BODY(B,12); POS_BODY(B,13); POS_BODY(B,14); POS_BODY(B,15); \
    POS_BODY(B,16); POS_BODY(B,17); POS_BODY(B,18); POS_BODY(B,19); \
    POS_BODY(B,20); POS_BODY(B,21); POS_BODY(B,22); POS_BODY(B,23); \
    POS_BODY(B,24); POS_BODY(B,25); POS_BODY(B,26); POS_BODY(B,27); \
    POS_BODY(B,28); POS_BODY(B,29); POS_BODY(B,30); POS_BODY(B,31);

#define SCAN_FN(NAME, B)                                                                \
static uint64_t NAME(const uint32_t *planar, size_t blocks, uint32_t target) {          \
    const uint32x4_t mvec = vdupq_n_u32(mask_b(B));                                      \
    const uint32x4_t tgt  = vdupq_n_u32(target & mask_b(B));                             \
    uint64_t count = 0;                                                                  \
    for (size_t blk = 0; blk < blocks; blk++) {                                          \
        const uint32_t *base = planar + blk * (size_t)(B) * FL_LANES;                    \
        __builtin_prefetch(base + (size_t)(B) * FL_LANES, 0, 0);                         \
        __builtin_prefetch(base + (size_t)(B) * FL_LANES + 16*FL_LANES, 0, 0);          \
        DECL_ACC                                                                         \
        BLOCK32(B)                                                                       \
        count += reduce8(a0,a1,a2,a3,a4,a5,a6,a7);                                       \
    }                                                                                    \
    return count;                                                                        \
}

SCAN_FN(scan_b3,  3)
SCAN_FN(scan_b5,  5)
SCAN_FN(scan_b7,  7)
SCAN_FN(scan_b8,  8)
SCAN_FN(scan_b12, 12)
SCAN_FN(scan_b16, 16)
SCAN_FN(scan_b24, 24)
SCAN_FN(scan_b32, 32)

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    switch (b) {
        case 3:  return scan_b3 (planar, blocks, target);
        case 5:  return scan_b5 (planar, blocks, target);
        case 7:  return scan_b7 (planar, blocks, target);
        case 8:  return scan_b8 (planar, blocks, target);
        case 12: return scan_b12(planar, blocks, target);
        case 16: return scan_b16(planar, blocks, target);
        case 24: return scan_b24(planar, blocks, target);
        case 32: return scan_b32(planar, blocks, target);
        default: return scan_general(planar, blocks, b, target);
    }
}
