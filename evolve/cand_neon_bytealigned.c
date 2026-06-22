/* cand_neon_v2.c — Apple M4 (ARM NEON) FastLanes scan, full per-b NEON.
 *
 * Adds NEON specialization for ALL widths 1..31 (not just 8/16/32). For each
 * compile-time b, every output position p has a compile-time (wp,off); the 32
 * contiguous lanes are 8 NEON vectors of 4x u32. off==0 -> mask+compare;
 * off!=0 -> funnel shift (srli|slli) with immediate shifts, then mask+compare.
 * Four u32 accumulator banks per block expose ILP. b in {8,16,32} keep the
 * byte/halfword/word-element fast paths.
 */
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

/* ---- b == 8 : byte-element compare, 16 values/vector ---- */
static uint64_t scan_b8(const uint8_t *base8, size_t blocks, uint8_t target) {
    const uint32_t bytes = 8u * 32u * 4u;            /* 1024 bytes/block */
    uint8x16_t tgt  = vdupq_n_u8(target);
    uint8x16_t ones = vdupq_n_u8(1);
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

/* ---- b == 16 : halfword-element compare, 8 values/vector ---- */
static uint64_t scan_b16(const uint16_t *base16, size_t blocks, uint16_t target) {
    const uint32_t hw = 16u * 32u * 2u;              /* 1024 halfwords/block */
    uint16x8_t tgt  = vdupq_n_u16(target);
    uint16x8_t ones = vdupq_n_u16(1);
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

/* ---- b == 32 : word-element compare, 4 banks for ILP ---- */
static uint64_t scan_b32(const uint32_t *planar, size_t blocks, uint32_t target) {
    const uint32_t words = 32u * 32u;
    uint32x4_t tgt  = vdupq_n_u32(target);
    uint32x4_t ones = vdupq_n_u32(1);
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

/* ============================================================================
 * Per-b NEON generic specialization (widths 1..31 except 8/16). Compile-time b
 * makes every off/wp/shift a constant. 32 lanes = 8 u32 vectors; 4 accumulator
 * banks. Counts are <= 32/block per lane so u32 accumulators (across all blocks)
 * could overflow only after 2^27 blocks; we widen per block to be safe & cheap.
 * ==========================================================================*/
/* aligned position: off==0, no shift */
#define POS_ALIGNED(p)                                                           \
    do {                                                                        \
        const unsigned bit = (p) * B, wp = bit >> 5;                            \
        const uint32_t *pw = base + (size_t)wp * FL_LANES;                      \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t v = vandq_u32(vld1q_u32(pw + q*4), vmask);               \
            acc[q] = vaddq_u32(acc[q], vandq_u32(vceqq_u32(v, vtgt), vone));    \
        }                                                                       \
    } while (0)

/* crossing position: off!=0, funnel shift with literal SH (=off) */
#define POS_CROSS(p, SH)                                                         \
    do {                                                                        \
        const unsigned bit = (p) * B, wp = bit >> 5;                            \
        const uint32_t *pw = base + (size_t)wp * FL_LANES;                      \
        const uint32_t *pw1 = pw + FL_LANES;                                    \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw  + q*4);                               \
            uint32x4_t hi = vld1q_u32(pw1 + q*4);                               \
            uint32x4_t v  = vorrq_u32(vshrq_n_u32(lo, (SH)),                    \
                                      vshlq_n_u32(hi, 32 - (SH)));              \
            v = vandq_u32(v, vmask);                                            \
            acc[q] = vaddq_u32(acc[q], vandq_u32(vceqq_u32(v, vtgt), vone));    \
        }                                                                       \
    } while (0)

/* dispatch one position by its compile-time off (1..31 literals) */
#define POS_NEON(p)                                                             \
    do {                                                                        \
        switch (((p) * B) & 31) {                                               \
            case 0:  POS_ALIGNED(p);     break;                                 \
            case 1:  POS_CROSS(p, 1);    break;                                 \
            case 2:  POS_CROSS(p, 2);    break;                                 \
            case 3:  POS_CROSS(p, 3);    break;                                 \
            case 4:  POS_CROSS(p, 4);    break;                                 \
            case 5:  POS_CROSS(p, 5);    break;                                 \
            case 6:  POS_CROSS(p, 6);    break;                                 \
            case 7:  POS_CROSS(p, 7);    break;                                 \
            case 8:  POS_CROSS(p, 8);    break;                                 \
            case 9:  POS_CROSS(p, 9);    break;                                 \
            case 10: POS_CROSS(p, 10);   break;                                 \
            case 11: POS_CROSS(p, 11);   break;                                 \
            case 12: POS_CROSS(p, 12);   break;                                 \
            case 13: POS_CROSS(p, 13);   break;                                 \
            case 14: POS_CROSS(p, 14);   break;                                 \
            case 15: POS_CROSS(p, 15);   break;                                 \
            case 16: POS_CROSS(p, 16);   break;                                 \
            case 17: POS_CROSS(p, 17);   break;                                 \
            case 18: POS_CROSS(p, 18);   break;                                 \
            case 19: POS_CROSS(p, 19);   break;                                 \
            case 20: POS_CROSS(p, 20);   break;                                 \
            case 21: POS_CROSS(p, 21);   break;                                 \
            case 22: POS_CROSS(p, 22);   break;                                 \
            case 23: POS_CROSS(p, 23);   break;                                 \
            case 24: POS_CROSS(p, 24);   break;                                 \
            case 25: POS_CROSS(p, 25);   break;                                 \
            case 26: POS_CROSS(p, 26);   break;                                 \
            case 27: POS_CROSS(p, 27);   break;                                 \
            case 28: POS_CROSS(p, 28);   break;                                 \
            case 29: POS_CROSS(p, 29);   break;                                 \
            case 30: POS_CROSS(p, 30);   break;                                 \
            case 31: POS_CROSS(p, 31);   break;                                 \
        }                                                                       \
    } while (0)

#define DEFINE_NEON_SCAN(B)                                                     \
    static uint64_t fl_scan_b##B(const uint32_t *planar, size_t blocks,        \
                                 uint32x4_t vmask, uint32x4_t vtgt) {          \
        uint32x4_t vone = vdupq_n_u32(1);                                       \
        uint64_t count = 0;                                                     \
        for (size_t blk = 0; blk < blocks; blk++) {                            \
            const uint32_t *base = planar + blk * (size_t)(B) * FL_LANES;       \
            uint32x4_t acc[8];                                                  \
            for (int q = 0; q < 8; q++) acc[q] = vdupq_n_u32(0);                \
            POS_NEON(0);  POS_NEON(1);  POS_NEON(2);  POS_NEON(3);              \
            POS_NEON(4);  POS_NEON(5);  POS_NEON(6);  POS_NEON(7);              \
            POS_NEON(8);  POS_NEON(9);  POS_NEON(10); POS_NEON(11);             \
            POS_NEON(12); POS_NEON(13); POS_NEON(14); POS_NEON(15);            \
            POS_NEON(16); POS_NEON(17); POS_NEON(18); POS_NEON(19);            \
            POS_NEON(20); POS_NEON(21); POS_NEON(22); POS_NEON(23);            \
            POS_NEON(24); POS_NEON(25); POS_NEON(26); POS_NEON(27);            \
            POS_NEON(28); POS_NEON(29); POS_NEON(30); POS_NEON(31);            \
            uint32x4_t s = vaddq_u32(vaddq_u32(vaddq_u32(acc[0],acc[1]),        \
                                               vaddq_u32(acc[2],acc[3])),       \
                                     vaddq_u32(vaddq_u32(acc[4],acc[5]),        \
                                               vaddq_u32(acc[6],acc[7])));      \
            count += vaddvq_u32(s);                                             \
        }                                                                       \
        return count;                                                           \
    }

#define B 1
DEFINE_NEON_SCAN(1)
#undef B
#define B 2
DEFINE_NEON_SCAN(2)
#undef B
#define B 3
DEFINE_NEON_SCAN(3)
#undef B
#define B 4
DEFINE_NEON_SCAN(4)
#undef B
#define B 5
DEFINE_NEON_SCAN(5)
#undef B
#define B 6
DEFINE_NEON_SCAN(6)
#undef B
#define B 7
DEFINE_NEON_SCAN(7)
#undef B
#define B 9
DEFINE_NEON_SCAN(9)
#undef B
#define B 10
DEFINE_NEON_SCAN(10)
#undef B
#define B 11
DEFINE_NEON_SCAN(11)
#undef B
#define B 12
DEFINE_NEON_SCAN(12)
#undef B
#define B 13
DEFINE_NEON_SCAN(13)
#undef B
#define B 14
DEFINE_NEON_SCAN(14)
#undef B
#define B 15
DEFINE_NEON_SCAN(15)
#undef B
#define B 17
DEFINE_NEON_SCAN(17)
#undef B
#define B 18
DEFINE_NEON_SCAN(18)
#undef B
#define B 19
DEFINE_NEON_SCAN(19)
#undef B
#define B 20
DEFINE_NEON_SCAN(20)
#undef B
#define B 21
DEFINE_NEON_SCAN(21)
#undef B
#define B 22
DEFINE_NEON_SCAN(22)
#undef B
#define B 23
DEFINE_NEON_SCAN(23)
#undef B
#define B 24
DEFINE_NEON_SCAN(24)
#undef B
#define B 25
DEFINE_NEON_SCAN(25)
#undef B
#define B 26
DEFINE_NEON_SCAN(26)
#undef B
#define B 27
DEFINE_NEON_SCAN(27)
#undef B
#define B 28
DEFINE_NEON_SCAN(28)
#undef B
#define B 29
DEFINE_NEON_SCAN(29)
#undef B
#define B 30
DEFINE_NEON_SCAN(30)
#undef B
#define B 31
DEFINE_NEON_SCAN(31)
#undef B

uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {
    size_t blocks = n / FL_BLOCK;
    if (b == 0) return 0;
    uint32_t m = mask_b(b);
    target &= m;
    uint32x4_t vmask = vdupq_n_u32(m);
    uint32x4_t vtgt  = vdupq_n_u32(target);
    switch (b) {
        case 1:  return fl_scan_b1 (planar, blocks, vmask, vtgt);
        case 2:  return fl_scan_b2 (planar, blocks, vmask, vtgt);
        case 3:  return fl_scan_b3 (planar, blocks, vmask, vtgt);
        case 4:  return fl_scan_b4 (planar, blocks, vmask, vtgt);
        case 5:  return fl_scan_b5 (planar, blocks, vmask, vtgt);
        case 6:  return fl_scan_b6 (planar, blocks, vmask, vtgt);
        case 7:  return fl_scan_b7 (planar, blocks, vmask, vtgt);
        case 8:  return scan_b8 ((const uint8_t  *)planar, blocks, (uint8_t)target);
        case 9:  return fl_scan_b9 (planar, blocks, vmask, vtgt);
        case 10: return fl_scan_b10(planar, blocks, vmask, vtgt);
        case 11: return fl_scan_b11(planar, blocks, vmask, vtgt);
        case 12: return fl_scan_b12(planar, blocks, vmask, vtgt);
        case 13: return fl_scan_b13(planar, blocks, vmask, vtgt);
        case 14: return fl_scan_b14(planar, blocks, vmask, vtgt);
        case 15: return fl_scan_b15(planar, blocks, vmask, vtgt);
        case 16: return scan_b16((const uint16_t *)planar, blocks, (uint16_t)target);
        case 17: return fl_scan_b17(planar, blocks, vmask, vtgt);
        case 18: return fl_scan_b18(planar, blocks, vmask, vtgt);
        case 19: return fl_scan_b19(planar, blocks, vmask, vtgt);
        case 20: return fl_scan_b20(planar, blocks, vmask, vtgt);
        case 21: return fl_scan_b21(planar, blocks, vmask, vtgt);
        case 22: return fl_scan_b22(planar, blocks, vmask, vtgt);
        case 23: return fl_scan_b23(planar, blocks, vmask, vtgt);
        case 24: return fl_scan_b24(planar, blocks, vmask, vtgt);
        case 25: return fl_scan_b25(planar, blocks, vmask, vtgt);
        case 26: return fl_scan_b26(planar, blocks, vmask, vtgt);
        case 27: return fl_scan_b27(planar, blocks, vmask, vtgt);
        case 28: return fl_scan_b28(planar, blocks, vmask, vtgt);
        case 29: return fl_scan_b29(planar, blocks, vmask, vtgt);
        case 30: return fl_scan_b30(planar, blocks, vmask, vtgt);
        case 31: return fl_scan_b31(planar, blocks, vmask, vtgt);
        case 32: return scan_b32(planar, blocks, target);
        default: return 0;
    }
}
