/* roofline.h — empirical memory-bandwidth ceiling (beta).
 *
 * Two ceilings, both reported:
 *   - STREAM-Triad (McCalpin): a[i] = b[i] + s*c[i]  -> sustainable read+write bandwidth.
 *   - scan-copy max: a sequential read-dominated memcpy-style sweep -> the achievable
 *     ceiling for a *read-mostly* scan, which is the honest denominator for f = scan/ beta.
 *
 * All buffers are sized >> LLC so we measure DRAM, not cache.
 */
#ifndef BBS_ROOFLINE_H
#define BBS_ROOFLINE_H

#include <stddef.h>

/* returns sustained GB/s (1e9 bytes/s). n_elems = number of doubles per array. */
double stream_triad_gbps(size_t n_elems, int repeats);

/* sequential read bandwidth: sum a huge uint8 buffer. read-only ceiling for scans. */
double scan_read_gbps(size_t n_bytes, int repeats);

#endif
