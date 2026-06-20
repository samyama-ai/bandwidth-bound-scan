/* codecs.h — light columnar encodings + fused decode-filter-aggregate scan kernels.
 *
 * Workhorse = bit-packing (arbitrary width b). FOR and DICT reduce to a bit-packed
 * code stream + a constant remap of the predicate target, so they reuse the bitpack
 * scan in "code space". RLE is a separate run-iterating scan.
 *
 * A "scan" here is the bandwidth-bound pipeline the problem describes: read the
 * compressed bytes, decode, evaluate an equality predicate, aggregate (count matches).
 * Two predicate styles isolate the branch-misprediction axis:
 *   - branchy:     if (v == target) count++;          (data-dependent branch)
 *   - branchfree:  count += (v == target);            (always-on, vectorizable)
 */
#ifndef BBS_CODECS_H
#define BBS_CODECS_H

#include <stdint.h>
#include <stddef.h>

/* ---- bit-packing (little-endian bit order into a uint32 word stream) ---- */
/* packed must hold at least (n*b + 31)/32 uint32 words. values must be < 2^b. */
void bp_encode(uint32_t *packed, const uint32_t *values, size_t n, unsigned b);
/* reference scalar decode (correctness oracle) */
void bp_decode(uint32_t *out, const uint32_t *packed, size_t n, unsigned b);
static inline size_t bp_words(size_t n, unsigned b) { return (n * (size_t)b + 31) / 32; }
static inline size_t bp_bytes(size_t n, unsigned b) { return bp_words(n, b) * 4; }

/* fused scans over a bit-packed stream: count values == target.
 * SCALAR = naive serial decode (variable shift; the compute-bound baseline).
 * VEC    = chunk-of-32 constant-shift decode (auto-vectorizable; FastLanes-style). */
uint64_t bp_scan_branchy(const uint32_t *packed, size_t n, unsigned b, uint32_t target);
uint64_t bp_scan_branchfree(const uint32_t *packed, size_t n, unsigned b, uint32_t target);
uint64_t bp_scan_vec_branchy(const uint32_t *packed, size_t n, unsigned b, uint32_t target);
uint64_t bp_scan_vec_branchfree(const uint32_t *packed, size_t n, unsigned b, uint32_t target);

/* ---- FastLanes-style bit-TRANSPOSED layout (lane-parallel, SIMD-decodable) ----
 * Values are stored in 1024-value blocks across 32 lanes (lane l holds values
 * l, l+32, ..., l+992, bit-packed). For a given output position all 32 lanes share the
 * SAME (word,shift), so unpacking is one vector shift+mask with no cross-lane gather --
 * this is what crosses sub-byte widths from compute-bound to bandwidth-bound.
 * Requires n % 1024 == 0 (the harness pads/uses divisible N). */
#define FL_LANES 32
#define FL_BLOCK 1024
void fl_encode(uint32_t *planar, const uint32_t *values, size_t n, unsigned b);
void fl_decode(uint32_t *out, const uint32_t *planar, size_t n, unsigned b);
uint64_t fl_scan_branchfree(const uint32_t *planar, size_t n, unsigned b, uint32_t target);
uint64_t fl_scan_branchy(const uint32_t *planar, size_t n, unsigned b, uint32_t target);

/* ---- materializing SELECT (writes matching positions) — isolates branch misprediction ----
 * Operates on a plain uint32 column (b=32, decode-free) so the branch, not decode, dominates.
 * branchy:     if (v==target) out[k++] = i;     (data-dependent store; mispredicts ~ sigma(1-sigma))
 * predicated:  out[k] = i; k += (v==target);    (always-store, branch-free)
 * Returns k (number selected). out must hold n entries. */
uint64_t sel32_branchy(const uint32_t *v, size_t n, uint32_t target, uint32_t *out);
uint64_t sel32_predicated(const uint32_t *v, size_t n, uint32_t target, uint32_t *out);

/* ---- RLE: (value,runlen) pairs ---- */
typedef struct { uint32_t value; uint32_t runlen; } rle_run_t;
/* returns number of runs written; runs must hold up to n entries. */
size_t rle_encode(rle_run_t *runs, const uint32_t *values, size_t n);
uint64_t rle_scan(const rle_run_t *runs, size_t nruns, uint32_t target);
static inline size_t rle_bytes(size_t nruns) { return nruns * sizeof(rle_run_t); }

/* ---- zone-map skipping over fixed-size blocks of a bit-packed column ---- */
/* per-block [min,max]; scan only blocks whose range can contain target. */
typedef struct { uint32_t mn, mx; } zone_t;
void zone_build(zone_t *zones, const uint32_t *values, size_t n, size_t block);
/* returns (count, *bytes_read) where bytes_read counts only scanned blocks' packed bytes. */
uint64_t zone_scan(const uint32_t *packed, const zone_t *zones, size_t n, size_t block,
                   unsigned b, uint32_t target, int branchfree, size_t *bytes_read);

#endif
