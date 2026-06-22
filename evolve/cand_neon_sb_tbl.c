/* cand_neon_sb_tbl.c — Apple M4 (ARM NEON) FastLanes scan, sub-byte focus.
 *
 * STRATEGY (the "byte-shuffle / fewer-ops-per-field" diversity contrast).
 *
 * The prior best (cand_neon_bytealigned/vecpos) dispatched every output
 * position into exactly two cases: off==0 (aligned) or off!=0 (a FULL funnel
 * shift: load lo, load hi, shr lo, shl hi, OR). But for sub-byte widths almost
 * every position with off!=0 does NOT actually straddle a 32-bit word boundary
 * (off+b <= 32). For those a single word holds the whole field, so the hi load,
 * the hi shift and the OR are pure waste. Counting per width:
 *   b : aligned / shift-only(no straddle) / true-cross
 *   1 :   1 / 31 / 0     5 :   1 / 27 / 4
 *   2 :   2 / 30 / 0     6 :   2 / 26 / 4
 *   3 :   1 / 29 / 2     7 :   1 / 25 / 6
 *   4 :   4 / 28 / 0
 * So 25..31 of 32 positions can drop from the 5-op funnel to a 2-op (shr+and)
 * extraction. That is the op-count cut: the common position goes from
 *   load,load,shr,shl,orr,and,ceq,sub  (8) -> load,shr,and,ceq,sub (5),
 * and we keep only one load live instead of two, halving L1 pressure.
 *
 * Compare/accumulate: vceqq_u32 gives 0/0xFFFFFFFF; subtract it from the
 * accumulator (vsubq_u32) to add +1 per match. 8 uint32x4 accumulator banks
 * (one per quarter of the 32 contiguous lanes) keep the M4 backend saturated.
 *
 * Correctness for all b=1..31 via compile-time per-b specialization (off, wp
 * and the straddle classification all fold to constants); b8/16/32 use
 * narrow-element compares; a portable scalar fallback covers anything else.
 */
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* aligned: off==0, field = lo & mask. */
#define POS_ALIGNED(WP)                                                         \
    do {                                                                        \
        const uint32_t *pw = base + (size_t)(WP) * FL_LANES;                    \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw + q * 4);                              \
            uint32x4_t v  = vandq_u32(lo, vmask);                               \
            acc[q] = vsubq_u32(acc[q], vceqq_u32(v, vtarget));                  \
        }                                                                       \
    } while (0)

/* shift-only: off!=0 but off+b<=32. Instead of shifting the data right by off
 * and masking, compare the field IN PLACE: (lo & (mask<<off)) == (target<<off).
 * mask<<off and target<<off are compile-time constants, so this drops the
 * per-position shift entirely -> load, and, ceq, sub (4 ops). */
#define POS_SHIFT(WP, OFF)                                                      \
    do {                                                                        \
        const uint32_t *pw = base + (size_t)(WP) * FL_LANES;                    \
        const uint32x4_t vm = vdupq_n_u32(M << (OFF));                          \
        const uint32x4_t vt = vdupq_n_u32(T << (OFF));                          \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw + q * 4);                              \
            uint32x4_t v  = vandq_u32(lo, vm);                                  \
            acc[q] = vsubq_u32(acc[q], vceqq_u32(v, vt));                       \
        }                                                                       \
    } while (0)

/* true cross: off!=0 and off+b>32, funnel shift across two words. */
#define POS_CROSS(WP, OFF)                                                      \
    do {                                                                        \
        const uint32_t *pw  = base + (size_t)(WP) * FL_LANES;                   \
        const uint32_t *pw1 = pw + FL_LANES;                                    \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw  + q * 4);                             \
            uint32x4_t hi = vld1q_u32(pw1 + q * 4);                             \
            uint32x4_t v  = vorrq_u32(vshrq_n_u32(lo, (OFF)),                   \
                                      vshlq_n_u32(hi, (32 - (OFF))));           \
            v = vandq_u32(v, vmask);                                            \
            acc[q] = vsubq_u32(acc[q], vceqq_u32(v, vtarget));                  \
        }                                                                       \
    } while (0)

/* Dispatch one position by its compile-time (off, straddle). __builtin_choose_expr
 * instantiates exactly one branch so the immediate shift count is always a literal. */
#define POS(p)                                                                  \
    do {                                                                        \
        enum { bit = (unsigned)(p) * B, wp = bit >> 5, off = bit & 31 };        \
        __builtin_choose_expr(off == 0,                                         \
            ({ POS_ALIGNED(wp); 0; }),                                          \
        __builtin_choose_expr((off + B) <= 32,                                  \
            ({ POS_SHIFT(wp, (off == 0 ? 1 : off)); 0; }),                      \
            ({ POS_CROSS(wp, (off == 0 ? 1 : off)); 0; })));                    \
    } while (0)

#define DEFINE_SCAN(B)                                                          \
    static uint64_t fl_scan_b##B(const uint32_t *planar, size_t blocks,        \
                                 uint32x4_t vmask, uint32x4_t vtarget,         \
                                 uint32_t T) {                                  \
        const uint32_t M = ((B) >= 32) ? 0xFFFFFFFFu : ((1u << (B)) - 1u);      \
        uint64_t count = 0;                                                     \
        const uint32x4_t z = vdupq_n_u32(0);                                    \
        for (size_t blk = 0; blk < blocks; blk++) {                            \
            const uint32_t *base = planar + blk * (size_t)(B) * FL_LANES;       \
            uint32x4_t acc[8] = { z, z, z, z, z, z, z, z };                     \
            POS(0);  POS(1);  POS(2);  POS(3);  POS(4);  POS(5);  POS(6);  POS(7);  \
            POS(8);  POS(9);  POS(10); POS(11); POS(12); POS(13); POS(14); POS(15); \
            POS(16); POS(17); POS(18); POS(19); POS(20); POS(21); POS(22); POS(23); \
            POS(24); POS(25); POS(26); POS(27); POS(28); POS(29); POS(30); POS(31); \
            uint32x4_t s = vaddq_u32(vaddq_u32(vaddq_u32(acc[0], acc[1]),       \
                                               vaddq_u32(acc[2], acc[3])),      \
                                     vaddq_u32(vaddq_u32(acc[4], acc[5]),       \
                                               vaddq_u32(acc[6], acc[7])));     \
            count += vaddvq_u32(s);                                             \
        }                                                                      \
        return count;                                                          \
    }

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

/* ---- b == 8 : byte-element compare, 16 values/vector ---- */
static uint64_t scan_b8(const uint8_t *base8, size_t blocks, uint8_t target) {
    const uint32_t bytes = 8u * 32u * 4u;
    uint8x16_t tgt = vdupq_n_u8(target), ones = vdupq_n_u8(1);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint8_t *base = base8 + blk * (size_t)bytes;
        uint16x8_t acc16 = vdupq_n_u16(0);
        for (uint32_t w = 0; w < bytes; w += 64) {
            uint8x16_t m0 = vandq_u8(vceqq_u8(vld1q_u8(base + w +  0), tgt), ones);
            uint8x16_t m1 = vandq_u8(vceqq_u8(vld1q_u8(base + w + 16), tgt), ones);
            uint8x16_t m2 = vandq_u8(vceqq_u8(vld1q_u8(base + w + 32), tgt), ones);
            uint8x16_t m3 = vandq_u8(vceqq_u8(vld1q_u8(base + w + 48), tgt), ones);
            uint8x16_t s  = vaddq_u8(vaddq_u8(m0, m1), vaddq_u8(m2, m3));
            acc16 = vpadalq_u8(acc16, s);
        }
        count += vaddvq_u16(acc16);
    }
    return count;
}

/* ---- b == 16 : halfword-element compare ---- */
static uint64_t scan_b16(const uint16_t *base16, size_t blocks, uint16_t target) {
    const uint32_t hw = 16u * 32u * 2u;
    uint16x8_t tgt = vdupq_n_u16(target), ones = vdupq_n_u16(1);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint16_t *base = base16 + blk * (size_t)hw;
        uint16x8_t a0 = vdupq_n_u16(0), a1 = vdupq_n_u16(0);
        for (uint32_t w = 0; w < hw; w += 32) {
            a0 = vaddq_u16(a0, vandq_u16(vceqq_u16(vld1q_u16(base + w +  0), tgt), ones));
            a1 = vaddq_u16(a1, vandq_u16(vceqq_u16(vld1q_u16(base + w +  8), tgt), ones));
            a0 = vaddq_u16(a0, vandq_u16(vceqq_u16(vld1q_u16(base + w + 16), tgt), ones));
            a1 = vaddq_u16(a1, vandq_u16(vceqq_u16(vld1q_u16(base + w + 24), tgt), ones));
        }
        count += (uint64_t)vaddlvq_u16(a0) + (uint64_t)vaddlvq_u16(a1);
    }
    return count;
}

/* ---- b == 32 : word-element compare, 4 banks ---- */
static uint64_t scan_b32(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 32u * 32u;
    uint32x4_t tgt = vdupq_n_u32(target), ones = vdupq_n_u32(1);
    uint64_t count = 0;
    for (size_t blk = 0; blk < blocks; blk++) {
        const uint32_t *base = planar + blk * (size_t)32 * FL_LANES;
        uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0);
        uint32x4_t a2 = vdupq_n_u32(0), a3 = vdupq_n_u32(0);
        for (uint32_t w = 0; w < words; w += 16) {
            a0 = vaddq_u32(a0, vandq_u32(vceqq_u32(vld1q_u32(base + w +  0), tgt), ones));
            a1 = vaddq_u32(a1, vandq_u32(vceqq_u32(vld1q_u32(base + w +  4), tgt), ones));
            a2 = vaddq_u32(a2, vandq_u32(vceqq_u32(vld1q_u32(base + w +  8), tgt), ones));
            a3 = vaddq_u32(a3, vandq_u32(vceqq_u32(vld1q_u32(base + w + 12), tgt), ones));
        }
        uint32x4_t s = vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3));
        count += vaddvq_u32(s);
    }
    return count;
}

/* Portable correct fallback (the seed) for any width not specialized. */
static uint64_t fl_scan_fallback(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK; uint32_t m = mask_b(b); uint64_t count = 0;
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

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    if (b == 0) return 0;
    uint32_t m = mask_b(b);
    target &= m;
    uint32x4_t vmask = vdupq_n_u32(m);
    uint32x4_t vtgt  = vdupq_n_u32(target);
    switch (b) {
        case 1:  return fl_scan_b1 (planar, blocks, vmask, vtgt, target);
        case 2:  return fl_scan_b2 (planar, blocks, vmask, vtgt, target);
        case 3:  return fl_scan_b3 (planar, blocks, vmask, vtgt, target);
        case 4:  return fl_scan_b4 (planar, blocks, vmask, vtgt, target);
        case 5:  return fl_scan_b5 (planar, blocks, vmask, vtgt, target);
        case 6:  return fl_scan_b6 (planar, blocks, vmask, vtgt, target);
        case 7:  return fl_scan_b7 (planar, blocks, vmask, vtgt, target);
        case 8:  return scan_b8 ((const uint8_t  *)planar, blocks, (uint8_t)target);
        case 9:  return fl_scan_b9 (planar, blocks, vmask, vtgt, target);
        case 10: return fl_scan_b10(planar, blocks, vmask, vtgt, target);
        case 11: return fl_scan_b11(planar, blocks, vmask, vtgt, target);
        case 12: return fl_scan_b12(planar, blocks, vmask, vtgt, target);
        case 13: return fl_scan_b13(planar, blocks, vmask, vtgt, target);
        case 14: return fl_scan_b14(planar, blocks, vmask, vtgt, target);
        case 15: return fl_scan_b15(planar, blocks, vmask, vtgt, target);
        case 16: return scan_b16((const uint16_t *)planar, blocks, (uint16_t)target);
        case 17: return fl_scan_b17(planar, blocks, vmask, vtgt, target);
        case 18: return fl_scan_b18(planar, blocks, vmask, vtgt, target);
        case 19: return fl_scan_b19(planar, blocks, vmask, vtgt, target);
        case 20: return fl_scan_b20(planar, blocks, vmask, vtgt, target);
        case 21: return fl_scan_b21(planar, blocks, vmask, vtgt, target);
        case 22: return fl_scan_b22(planar, blocks, vmask, vtgt, target);
        case 23: return fl_scan_b23(planar, blocks, vmask, vtgt, target);
        case 24: return fl_scan_fallback(planar, n, b, target);
        case 25: return fl_scan_b25(planar, blocks, vmask, vtgt, target);
        case 26: return fl_scan_b26(planar, blocks, vmask, vtgt, target);
        case 27: return fl_scan_b27(planar, blocks, vmask, vtgt, target);
        case 28: return fl_scan_b28(planar, blocks, vmask, vtgt, target);
        case 29: return fl_scan_b29(planar, blocks, vmask, vtgt, target);
        case 30: return fl_scan_b30(planar, blocks, vmask, vtgt, target);
        case 31: return fl_scan_b31(planar, blocks, vmask, vtgt, target);
        case 32: return scan_b32(planar, blocks, target);
        default: return fl_scan_fallback(planar, n, b, target);
    }
}
