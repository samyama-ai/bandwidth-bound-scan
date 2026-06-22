/* cand_neon_sb_mlp.c — Apple M4 (ARM NEON) FastLanes scan, sub-byte focused.
 *
 * Contract:
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target)
 *
 * Layout (FastLanes bit-transposed): block blk starts at planar + blk*b*32.
 * For output position p (0..31): bit=p*b, wp=bit>>5, off=bit&31. The 32 lanes of a
 * word row w are 8 contiguous uint32x4 vectors at base[w*32 + 4*q], q=0..7.
 *
 * Two structural wins over the per-position funnel-shift baseline (~14.2 v/ns):
 *
 *  (1) COMPARE-IN-PLACE.  Instead of funnel-shifting every position's value down
 *      to bit 0 and then `(v & m) == target`, we pre-shift the *constants*:
 *      for a position fully contained in one word (off + b <= 32) the test is
 *      `(word & (m<<off)) == (target<<off)` — ONE vand + ONE vceq, no data shift.
 *      For b<=7 only ~b of 32 positions cross a 32-bit boundary; the rest become
 *      2-op tests. Boundary-crossing positions still use a funnel shift.
 *
 *  (2) MULTI-BLOCK MLP/ILP.  Process UF independent FastLanes blocks per loop
 *      iteration, each with its own pool of 8 uint32x4 accumulators, so the M4's
 *      many NEON pipes and load/store units stay saturated and inter-block loads
 *      overlap. Software-prefetch the blocks UF ahead. All accumulators reduce
 *      once at the very end.
 *
 * Correct for all b=1..32; a scalar general fallback covers anything unspecialized.
 */
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* horizontal add of 8 chains */
static inline uint64_t reduce8(uint32x4_t a0, uint32x4_t a1, uint32x4_t a2, uint32x4_t a3,
                               uint32x4_t a4, uint32x4_t a5, uint32x4_t a6, uint32x4_t a7) {
    uint32x4_t s0 = vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3));
    uint32x4_t s1 = vaddq_u32(vaddq_u32(a4, a5), vaddq_u32(a6, a7));
    return (uint64_t)vaddvq_u32(vaddq_u32(s0, s1));
}

/* ---- General scalar fallback (identical semantics to the seed) ---- */
static uint64_t scan_general(const uint32_t *planar, size_t blocks, unsigned b, uint32_t target) {
    uint32_t m = mask_b(b);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)b * FL_LANES;
        uint32_t partial[FL_LANES] = {0};
        for (unsigned p = 0; p < 32; p++) {
            unsigned bit = p * b, wp = bit >> 5, off = bit & 31;
            const uint32_t *pw = base + wp * FL_LANES;
            if (off + b <= 32) {
                for (unsigned l = 0; l < FL_LANES; l++)
                    partial[l] += (((pw[l] >> off) & m) == target);
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

/* =====================================================================
 * Per-b NEON specialization, compare-in-place, UF blocks per iteration.
 *
 * For one position p we emit either a "contained" test (1 vand + 1 vceq,
 * compared against a pre-shifted target with a pre-shifted mask) or a
 * "crossing" funnel test. WP/OFF are compile-time constants. The `base`
 * pointer and accumulator bank are macro arguments so we can stamp out UF
 * independent block bodies that share the per-b constant vectors.
 * ===================================================================== */

/* contained position: off + b <= 32. test (word & MSHIFT) == TSHIFT.
   MSHIFT = m<<off, TSHIFT = target<<off (both runtime-constant vectors). */
#define POS_IN(BASE, WP, MS, TS, A0,A1,A2,A3,A4,A5,A6,A7)                          \
do {                                                                              \
    const uint32_t *pw = (BASE) + (size_t)(WP) * FL_LANES;                        \
    A0 = vsubq_u32(A0, vceqq_u32(vandq_u32(vld1q_u32(pw +  0), (MS)), (TS)));     \
    A1 = vsubq_u32(A1, vceqq_u32(vandq_u32(vld1q_u32(pw +  4), (MS)), (TS)));     \
    A2 = vsubq_u32(A2, vceqq_u32(vandq_u32(vld1q_u32(pw +  8), (MS)), (TS)));     \
    A3 = vsubq_u32(A3, vceqq_u32(vandq_u32(vld1q_u32(pw + 12), (MS)), (TS)));     \
    A4 = vsubq_u32(A4, vceqq_u32(vandq_u32(vld1q_u32(pw + 16), (MS)), (TS)));     \
    A5 = vsubq_u32(A5, vceqq_u32(vandq_u32(vld1q_u32(pw + 20), (MS)), (TS)));     \
    A6 = vsubq_u32(A6, vceqq_u32(vandq_u32(vld1q_u32(pw + 24), (MS)), (TS)));     \
    A7 = vsubq_u32(A7, vceqq_u32(vandq_u32(vld1q_u32(pw + 28), (MS)), (TS)));     \
} while (0)

/* crossing position: funnel shift by OFF, then mask & compare against tgt. */
#define POS_X(BASE, WP, OFF, MV, TG, A0,A1,A2,A3,A4,A5,A6,A7)                      \
do {                                                                              \
    const uint32_t *pw  = (BASE) + (size_t)(WP) * FL_LANES;                       \
    const uint32_t *pw1 = pw + FL_LANES;                                          \
    uint32x4_t v0 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+ 0),32-(OFF)),vld1q_u32(pw+ 0),(OFF)),(MV)); \
    uint32x4_t v1 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+ 4),32-(OFF)),vld1q_u32(pw+ 4),(OFF)),(MV)); \
    uint32x4_t v2 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+ 8),32-(OFF)),vld1q_u32(pw+ 8),(OFF)),(MV)); \
    uint32x4_t v3 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+12),32-(OFF)),vld1q_u32(pw+12),(OFF)),(MV)); \
    uint32x4_t v4 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+16),32-(OFF)),vld1q_u32(pw+16),(OFF)),(MV)); \
    uint32x4_t v5 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+20),32-(OFF)),vld1q_u32(pw+20),(OFF)),(MV)); \
    uint32x4_t v6 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+24),32-(OFF)),vld1q_u32(pw+24),(OFF)),(MV)); \
    uint32x4_t v7 = vandq_u32(vsraq_n_u32(vshlq_n_u32(vld1q_u32(pw1+28),32-(OFF)),vld1q_u32(pw+28),(OFF)),(MV)); \
    A0 = vsubq_u32(A0, vceqq_u32(v0,(TG))); A1 = vsubq_u32(A1, vceqq_u32(v1,(TG)));         \
    A2 = vsubq_u32(A2, vceqq_u32(v2,(TG))); A3 = vsubq_u32(A3, vceqq_u32(v3,(TG)));         \
    A4 = vsubq_u32(A4, vceqq_u32(v4,(TG))); A5 = vsubq_u32(A5, vceqq_u32(v5,(TG)));         \
    A6 = vsubq_u32(A6, vceqq_u32(v6,(TG))); A7 = vsubq_u32(A7, vceqq_u32(v7,(TG)));         \
} while (0)

/* one position: choose contained vs crossing at compile time.
   For a contained position the pre-shifted constants are msh[off]/tsh[off]. */
#define POS(B, p, BASE, A0,A1,A2,A3,A4,A5,A6,A7)                                   \
do {                                                                              \
    enum { bit_ = (p)*(B), wp_ = bit_ >> 5, off_ = bit_ & 31 };                   \
    if (off_ + (B) <= 32)                                                         \
        POS_IN(BASE, wp_, msh[off_], tsh[off_], A0,A1,A2,A3,A4,A5,A6,A7);         \
    else                                                                          \
        POS_X(BASE, wp_, (off_ ? off_ : 1), mvec, tgt, A0,A1,A2,A3,A4,A5,A6,A7);  \
} while (0)

/* all 32 positions of one block into one accumulator bank */
#define BLOCK32(B, BASE, A0,A1,A2,A3,A4,A5,A6,A7)                                  \
    POS(B,0,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   POS(B,1,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   \
    POS(B,2,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   POS(B,3,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   \
    POS(B,4,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   POS(B,5,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   \
    POS(B,6,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   POS(B,7,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   \
    POS(B,8,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   POS(B,9,BASE,A0,A1,A2,A3,A4,A5,A6,A7);   \
    POS(B,10,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,11,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,12,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,13,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,14,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,15,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,16,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,17,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,18,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,19,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,20,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,21,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,22,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,23,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,24,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,25,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,26,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,27,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,28,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,29,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  \
    POS(B,30,BASE,A0,A1,A2,A3,A4,A5,A6,A7);  POS(B,31,BASE,A0,A1,A2,A3,A4,A5,A6,A7);

/* Pre-shifted constant tables: msh[off]=m<<off, tsh[off]=(target&m)<<off. */
#define DECL_CONSTS(B)                                                             \
    const uint32_t m_ = mask_b(B);                                                 \
    const uint32_t t_ = target & m_;                                              \
    const uint32x4_t mvec = vdupq_n_u32(m_);                                       \
    const uint32x4_t tgt  = vdupq_n_u32(t_);                                       \
    uint32x4_t msh[32], tsh[32];                                                   \
    for (int s = 0; s < 32; s++) {                                                 \
        msh[s] = vdupq_n_u32((uint32_t)(m_ << s));                                 \
        tsh[s] = vdupq_n_u32((uint32_t)(t_ << s));                                 \
    }

#define ACC8(P) \
    uint32x4_t P##0=vdupq_n_u32(0),P##1=vdupq_n_u32(0),P##2=vdupq_n_u32(0),P##3=vdupq_n_u32(0), \
              P##4=vdupq_n_u32(0),P##5=vdupq_n_u32(0),P##6=vdupq_n_u32(0),P##7=vdupq_n_u32(0);

/* ---- UF = 4 blocks per iteration ---- */
#define SCAN_FN(NAME, B)                                                           \
static uint64_t NAME(const uint32_t *planar, size_t blocks, uint32_t target) {     \
    DECL_CONSTS(B)                                                                  \
    const size_t BW = (size_t)(B) * FL_LANES;  /* uint32 words per block */         \
    uint64_t count = 0;                                                            \
    ACC8(a) ACC8(c) ACC8(d) ACC8(e)                                                \
    size_t blk = 0;                                                                \
    for (; blk + 4 <= blocks; blk += 4) {                                          \
        const uint32_t *b0 = planar + (blk + 0) * BW;                              \
        const uint32_t *b1 = planar + (blk + 1) * BW;                              \
        const uint32_t *b2 = planar + (blk + 2) * BW;                              \
        const uint32_t *b3 = planar + (blk + 3) * BW;                              \
        __builtin_prefetch(b0 + 4 * BW, 0, 0);                                     \
        __builtin_prefetch(b0 + 4 * BW + 16 * FL_LANES, 0, 0);                     \
        __builtin_prefetch(b2 + 4 * BW, 0, 0);                                     \
        __builtin_prefetch(b2 + 4 * BW + 16 * FL_LANES, 0, 0);                     \
        BLOCK32(B, b0, a0,a1,a2,a3,a4,a5,a6,a7)                                     \
        BLOCK32(B, b1, c0,c1,c2,c3,c4,c5,c6,c7)                                     \
        BLOCK32(B, b2, d0,d1,d2,d3,d4,d5,d6,d7)                                     \
        BLOCK32(B, b3, e0,e1,e2,e3,e4,e5,e6,e7)                                     \
    }                                                                              \
    for (; blk < blocks; blk++) {                                                  \
        const uint32_t *b0 = planar + blk * BW;                                    \
        BLOCK32(B, b0, a0,a1,a2,a3,a4,a5,a6,a7)                                     \
    }                                                                              \
    count += reduce8(a0,a1,a2,a3,a4,a5,a6,a7);                                     \
    count += reduce8(c0,c1,c2,c3,c4,c5,c6,c7);                                     \
    count += reduce8(d0,d1,d2,d3,d4,d5,d6,d7);                                     \
    count += reduce8(e0,e1,e2,e3,e4,e5,e6,e7);                                     \
    return count;                                                                  \
}

SCAN_FN(scan_b1, 1)
SCAN_FN(scan_b2, 2)
SCAN_FN(scan_b3, 3)
SCAN_FN(scan_b4, 4)
SCAN_FN(scan_b5, 5)
SCAN_FN(scan_b6, 6)
SCAN_FN(scan_b7, 7)

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    switch (b) {
        case 1: return scan_b1(planar, blocks, target);
        case 2: return scan_b2(planar, blocks, target);
        case 3: return scan_b3(planar, blocks, target);
        case 4: return scan_b4(planar, blocks, target);
        case 5: return scan_b5(planar, blocks, target);
        case 6: return scan_b6(planar, blocks, target);
        case 7: return scan_b7(planar, blocks, target);
        default: return scan_general(planar, blocks, b, target);
    }
}
