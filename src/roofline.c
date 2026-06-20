#include "roofline.h"
#include "common.h"

double stream_triad_gbps(size_t n, int repeats) {
    double *a = (double *)aligned_alloc64(n * sizeof(double));
    double *b = (double *)aligned_alloc64(n * sizeof(double));
    double *c = (double *)aligned_alloc64(n * sizeof(double));
    if (!a || !b || !c) { free(a); free(b); free(c); return -1.0; }
    for (size_t i = 0; i < n; i++) { b[i] = 1.0; c[i] = 2.0; a[i] = 0.0; }
    const double s = 3.0;
    double best = 0.0;
    for (int r = 0; r < repeats; r++) {
        double t0 = now_ns();
        for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
        double t1 = now_ns();
        /* triad touches 3 arrays/elem: 2 read + 1 write = 24 bytes/elem */
        double gbps = (double)n * 24.0 / (t1 - t0); /* (bytes) / (ns) = GB/s */
        if (gbps > best) best = gbps;
    }
    /* prevent dead-code elimination */
    volatile double sink = a[n - 1]; (void)sink;
    free(a); free(b); free(c);
    return best;
}

double scan_read_gbps(size_t n_bytes, int repeats) {
    uint8_t *buf = (uint8_t *)aligned_alloc64(n_bytes);
    if (!buf) return -1.0;
    memset(buf, 1, n_bytes);
    double best = 0.0;
    for (int r = 0; r < repeats; r++) {
        uint64_t acc = 0;
        const uint64_t *p = (const uint64_t *)buf;
        size_t n8 = n_bytes / 8;
        double t0 = now_ns();
        for (size_t i = 0; i < n8; i++) acc += p[i];
        double t1 = now_ns();
        double gbps = (double)n_bytes / (t1 - t0);
        if (gbps > best) best = gbps;
        if (acc == 0xDEADBEEF) buf[0] ^= 1; /* defeat DCE */
    }
    free(buf);
    return best;
}
