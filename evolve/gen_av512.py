#!/usr/bin/env python3
"""Generate AVX-512 sub-byte FastLanes scan with per-b unroll factor (UF).

Usage: gen_av512.py "b1:uf,b2:uf,..."  e.g. "1:8,2:8,3:8,4:8,5:4,6:4,7:4"
Writes the kernel to stdout.
"""
import sys

HEADER = r'''/* cand_avx512_subbyte.c — AVX-512 FastLanes scan, sub-byte (b=1..7) focused.
 *
 * Contract:
 *   uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target)
 *
 * Layout (FastLanes bit-transposed): block blk starts at planar + blk*b*32.
 * For output position p (0..31): bit=p*b, wp=bit>>5, off=bit&31. The 32 lanes of a
 * word row w are TWO contiguous __m512i (16 uint32 each) at base[w*32 + 0/16].
 *
 *  (1) COMPARE-IN-PLACE: contained position (off+b<=32) -> (word & (m<<off)) ==
 *      (target<<off): ONE vpand + ONE vpcmpeqd_mask per 16-lane half, no data shift.
 *      Boundary-crossing positions use a funnel shift.
 *  (2) MULTI-BLOCK ILP: UF independent FastLanes blocks per iteration, each into its
 *      own scalar popcnt accumulator. UF is tuned per-b (see dispatch).
 *
 * Per-b specialization (compile-time off/wp -> immediate shifts). Correct for all
 * b=1..32; a scalar general fallback covers widths >7.
 */
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#define FL_LANES 32
#define FL_BLOCK 1024

static inline uint32_t mask_b(unsigned b) { return b >= 32 ? 0xFFFFFFFFu : ((1u << b) - 1u); }

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

#define POS_IN(BASE, WP, MS, TS, ACC)                                              \
do {                                                                              \
    const uint32_t *pw = (BASE) + (size_t)(WP) * FL_LANES;                        \
    __m512i w0 = _mm512_load_si512((const void *)(pw + 0));                       \
    __m512i w1 = _mm512_load_si512((const void *)(pw + 16));                      \
    __mmask16 k0 = _mm512_cmpeq_epi32_mask(_mm512_and_si512(w0, (MS)), (TS));     \
    __mmask16 k1 = _mm512_cmpeq_epi32_mask(_mm512_and_si512(w1, (MS)), (TS));     \
    (ACC) += (uint64_t)_mm_popcnt_u32((uint32_t)k0 | ((uint32_t)k1 << 16));       \
} while (0)

#define POS_X(BASE, WP, OFF, MV, TG, ACC)                                          \
do {                                                                              \
    const uint32_t *pw  = (BASE) + (size_t)(WP) * FL_LANES;                       \
    const uint32_t *pw1 = pw + FL_LANES;                                          \
    __m512i lo0 = _mm512_load_si512((const void *)(pw  + 0));                     \
    __m512i lo1 = _mm512_load_si512((const void *)(pw  + 16));                    \
    __m512i hi0 = _mm512_load_si512((const void *)(pw1 + 0));                     \
    __m512i hi1 = _mm512_load_si512((const void *)(pw1 + 16));                    \
    __m512i v0  = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi32(lo0,(OFF)),  \
                                   _mm512_slli_epi32(hi0,32-(OFF))), (MV));        \
    __m512i v1  = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi32(lo1,(OFF)),  \
                                   _mm512_slli_epi32(hi1,32-(OFF))), (MV));        \
    __mmask16 k0 = _mm512_cmpeq_epi32_mask(v0, (TG));                             \
    __mmask16 k1 = _mm512_cmpeq_epi32_mask(v1, (TG));                             \
    (ACC) += (uint64_t)_mm_popcnt_u32((uint32_t)k0 | ((uint32_t)k1 << 16));       \
} while (0)

#define POS(B, p, BASE, ACC)                                                       \
do {                                                                              \
    enum { bit_ = (p)*(B), wp_ = bit_ >> 5, off_ = bit_ & 31 };                   \
    if (off_ + (B) <= 32)                                                         \
        POS_IN(BASE, wp_, msh[off_], tsh[off_], ACC);                            \
    else                                                                          \
        POS_X(BASE, wp_, (off_ ? off_ : 1), mvec, tgt, ACC);                     \
} while (0)

#define BLOCK32(B, BASE, ACC)                                                      \
    POS(B,0,BASE,ACC);  POS(B,1,BASE,ACC);  POS(B,2,BASE,ACC);  POS(B,3,BASE,ACC); \
    POS(B,4,BASE,ACC);  POS(B,5,BASE,ACC);  POS(B,6,BASE,ACC);  POS(B,7,BASE,ACC); \
    POS(B,8,BASE,ACC);  POS(B,9,BASE,ACC);  POS(B,10,BASE,ACC); POS(B,11,BASE,ACC);\
    POS(B,12,BASE,ACC); POS(B,13,BASE,ACC); POS(B,14,BASE,ACC); POS(B,15,BASE,ACC);\
    POS(B,16,BASE,ACC); POS(B,17,BASE,ACC); POS(B,18,BASE,ACC); POS(B,19,BASE,ACC);\
    POS(B,20,BASE,ACC); POS(B,21,BASE,ACC); POS(B,22,BASE,ACC); POS(B,23,BASE,ACC);\
    POS(B,24,BASE,ACC); POS(B,25,BASE,ACC); POS(B,26,BASE,ACC); POS(B,27,BASE,ACC);\
    POS(B,28,BASE,ACC); POS(B,29,BASE,ACC); POS(B,30,BASE,ACC); POS(B,31,BASE,ACC);

#define DECL_CONSTS(B)                                                             \
    const uint32_t m_ = mask_b(B);                                                 \
    const uint32_t t_ = target & m_;                                              \
    const __m512i mvec = _mm512_set1_epi32((int)m_);                               \
    const __m512i tgt  = _mm512_set1_epi32((int)t_);                               \
    __m512i msh[32], tsh[32];                                                       \
    for (int s = 0; s < 32; s++) {                                                 \
        msh[s] = _mm512_set1_epi32((int)(m_ << s));                                 \
        tsh[s] = _mm512_set1_epi32((int)(t_ << s));                                 \
    }
'''

def emit_scan(b, uf):
    accs = ",".join("a%d=0" % i for i in range(uf))
    lines = []
    lines.append("static uint64_t scan_b%d(const uint32_t *planar, size_t blocks, uint32_t target) {" % b)
    lines.append("    DECL_CONSTS(%d)" % b)
    lines.append("    const size_t BW = (size_t)(%d) * FL_LANES;" % b)
    lines.append("    uint64_t %s;" % accs)
    lines.append("    size_t blk = 0;")
    lines.append("    for (; blk + %d <= blocks; blk += %d) {" % (uf, uf))
    for i in range(uf):
        lines.append("        BLOCK32(%d, planar + (blk+%d)*BW, a%d)" % (b, i, i))
    lines.append("    }")
    lines.append("    for (; blk < blocks; blk++) { BLOCK32(%d, planar + blk*BW, a0) }" % b)
    lines.append("    return %s;" % "+".join("a%d" % i for i in range(uf)))
    lines.append("}")
    return "\n".join(lines)

def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "1:8,2:8,3:8,4:8,5:4,6:4,7:4"
    ufmap = {}
    for part in spec.split(","):
        b, uf = part.split(":")
        ufmap[int(b)] = int(uf)
    out = [HEADER, ""]
    for b in range(1, 8):
        out.append(emit_scan(b, ufmap[b]))
        out.append("")
    out.append("uint64_t fl_scan_candidate(const uint32_t *planar, size_t n, unsigned b, uint32_t target) {")
    out.append("    size_t blocks = n / FL_BLOCK;")
    out.append("    switch (b) {")
    for b in range(1, 8):
        out.append("        case %d: return scan_b%d(planar, blocks, target);" % (b, b))
    out.append("        default: return scan_general(planar, blocks, b, target);")
    out.append("    }")
    out.append("}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
