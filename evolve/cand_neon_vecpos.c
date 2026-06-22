/* cand_neon_vecpos.c — Apple M4 (ARM NEON) port of the vecpos FastLanes scan.
 *
 * Key insight (same as x86 vecpos): for output position p, the 32 lanes of a
 * given word wp=(p*b)>>5 are CONTIGUOUS in memory at base[wp*32 + 0..31].
 * NEON is 128-bit = 4x uint32, so those 32 lanes = EIGHT contiguous uint32x4
 * loads (vld1q_u32). For off!=0 also load the high word and funnel-shift.
 *
 * Per-b specialization (switch on b -> compile-time B) makes off=(p*B)&31 and
 * wp compile-time constants, so the NEON shifts become immediate-count
 * vshrq_n_u32 / vshlq_n_u32 (required for constant shift amounts).
 *
 * Compare: vceqq_u32 yields 0/0xFFFFFFFF per lane; subtracting that mask from an
 * accumulator (vsubq_u32) adds +1 on a match. Reduce per block with vaddvq_u32.
 *
 * Eight uint32x4 accumulators (one per quarter of the 32 lanes) give wide ILP so
 * the M4's vector backend stays busy while loads stream in.
 */
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>

#define FL_LANES 32
#define FL_BLOCK 1024

/* One output position p, with B a compile-time constant.
 * off/wp fold to constants; shifts become immediate-count NEON shifts. */
/* off==0 path: pure load + mask + compare. */
#define POS_ALIGNED(WP)                                                         \
    do {                                                                        \
        const uint32_t *pw = base + (size_t)(WP) * FL_LANES;                    \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw + q * 4);                              \
            uint32x4_t v  = vandq_u32(lo, vmask);                              \
            acc[q] = vsubq_u32(acc[q], vceqq_u32(v, vtarget));                  \
        }                                                                       \
    } while (0)

/* off!=0 path: funnel-shift with literal immediates OFF (1..31). */
#define POS_CROSS(WP, OFF)                                                      \
    do {                                                                        \
        const uint32_t *pw  = base + (size_t)(WP) * FL_LANES;                   \
        const uint32_t *pw1 = pw + FL_LANES;                                    \
        for (int q = 0; q < 8; q++) {                                           \
            uint32x4_t lo = vld1q_u32(pw  + q * 4);                            \
            uint32x4_t hi = vld1q_u32(pw1 + q * 4);                            \
            uint32x4_t v  = vorrq_u32(vshrq_n_u32(lo, (OFF)),                   \
                                      vshlq_n_u32(hi, (32 - (OFF))));           \
            v = vandq_u32(v, vmask);                                            \
            acc[q] = vsubq_u32(acc[q], vceqq_u32(v, vtarget));                  \
        }                                                                       \
    } while (0)

/* Dispatch one position: __builtin_choose_expr keeps only the live branch, so
 * the immediate passed to the NEON shift is always a literal constant in 1..31
 * (the dead branch's OFF=0 shift is never instantiated). */
#define POS(p)                                                                  \
    do {                                                                        \
        enum { bit = (unsigned)(p) * B, wp = bit >> 5, off = bit & 31 };        \
        __builtin_choose_expr(off == 0,                                         \
            ({ POS_ALIGNED(wp); 0; }),                                          \
            ({ POS_CROSS(wp, (off == 0 ? 1 : off)); 0; }));                     \
    } while (0)

#define DEFINE_SCAN(B)                                                          \
    static uint64_t fl_scan_b##B(const uint32_t *planar, size_t blocks,        \
                                 uint32x4_t vmask, uint32x4_t vtarget) {        \
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
    uint32x4_t vmask   = vdupq_n_u32(mask_b(b));
    uint32x4_t vtarget = vdupq_n_u32(target);
    switch (b) {
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
        /* b24 (3-byte aligned): the scalar fallback autovectorizes better than
         * the double-load funnel-shift NEON path here, so route it through. */
        case 24: return fl_scan_fallback(planar, n, b, target);
        case 25: return fl_scan_b25(planar, blocks, vmask, vtarget);
        case 26: return fl_scan_b26(planar, blocks, vmask, vtarget);
        case 27: return fl_scan_b27(planar, blocks, vmask, vtarget);
        case 28: return fl_scan_b28(planar, blocks, vmask, vtarget);
        case 29: return fl_scan_b29(planar, blocks, vmask, vtarget);
        case 30: return fl_scan_b30(planar, blocks, vmask, vtarget);
        case 31: return fl_scan_b31(planar, blocks, vmask, vtarget);
        case 32: return fl_scan_b32(planar, blocks, vmask, vtarget);
        default: return fl_scan_fallback(planar, n, b, target);
    }
}
