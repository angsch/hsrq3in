#include "utils.h"

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

double get_time (void)
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}


int get_size_with_padding(const int n)
{
#if defined(__AVX512F__)
    // Round up n to the next multiple of 8.
    return 8 * ((n + 7) / 8);
#else
    // Round up n to the next multiple of 4.
    return 4 * ((n + 3) / 4);
#endif
}


void scale_tile(int m, int n, 
    double *restrict const X, int ldX, const scaling_t *beta)
{
#ifdef INTSCALING
    double alpha = ldexp(1.0, beta[0]);
#else
    double alpha = beta[0];
#endif

#define X(i,j) X[(i) + (j) * ldX]

    // Scale vector, if necessary.
    if (alpha != 1.0) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i,j) = alpha * X(i,j);
            }
        }
    }

#undef X
}


// Scale everything that is not enclosed in (ilo:ihi,jlo:jhi)
void scale_excluding(int m, int n, int ilo, int ihi, int jlo, int jhi,
    double *restrict const X, int ldX, const scaling_t *beta)
{
#ifdef INTSCALING
    double alpha = ldexp(1.0, beta[0]);
#else
    double alpha = beta[0];
#endif

#define X(i,j) X[(i) + (j) * ldX]

    // Scale vector, if necessary.
    if (alpha != 1.0) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if ((ilo <= i && i <= ihi) && (jlo <= j && j <= jhi)) {
                    // Skip. We are in the area that is to be spared.
                }
                else {
                    X(i,j) = alpha * X(i,j);
                }
            }
        }
    }

#undef X
}

// Scale the entries {0, 1, ..., n-1} \ {skip}
void scale_except(
    int n, int skip, double *restrict const x, const scaling_t * beta)
{
#ifdef INTSCALING
    double alpha = ldexp(1.0, beta[0]);
#else
    double alpha = beta[0];
#endif

    // Scale vector, if necessary.
    if (alpha != 1.0) {
        for (int i = 0; i < n; i++) {
            if (i == skip) {
                // Skip.
            }
            else {
                x[i] = alpha * x[i];
            }
        }
    }

}



void dgemm(
    const char transa, const char transb,
    const int m, const int n, const int k,
    const double alpha, const double *restrict const A, const int ldA,
    const double *restrict const B, const int ldB,
    const double beta, double *restrict const C, const int ldC)
{
    extern void dgemm_(
        const char *transa, const char *transb,
        const int *m, const int *n, const int *k,
        const double *alpha, const double *a, const int *lda,
        const double *b, const int *ldb,
        const double *beta, double *c, const int *ldc);

    dgemm_(&transa, &transb,
           &m, &n, &k,
           &alpha, A, &ldA,
           B, &ldB,
           &beta, C, &ldC);

}



void dgemv(
    const char trans,
    const int m, const int n,
    const double alpha, const double *restrict const A, const int ldA,
    const double *restrict x, int incx,
    const double beta, double *restrict const y, int incy)
{
    extern void dgemv_(
        const char *trans,
        const int *m, const int *n,
        const double *alpha, const double *a, const int *lda,
        const double *x, const int *incx,
        const double *beta, double *y, const int *incy);

    dgemv_(&trans,
           &m, &n,
           &alpha, A, &ldA,
           x, &incx,
           &beta, y, &incy);
}



double dlange(const char norm, const int m, const int n,
    const double *restrict const A, const int ldA)
{
    double *work = NULL;

    if (norm == 'I' || norm == 'i') {
        work = (double *) malloc(m * sizeof(double));
    }

    extern double dlange_(const char *norm, const int *m, const int *n,
        const double *a, const int *ldA, double *work);

    double nrm = dlange_(&norm, &m, &n, A, &ldA, work);

    if (norm == 'I' || norm == 'i') {
        free(work);
    }

    return nrm;
}


void dlaln2(int ltrans, int na, int nw, double smin, double ca, double *A,
    int ldA, double d1, double d2, double *B, int ldB, double wr, double wi,
    double *X, int ldX, double *scale, double *xnorm)
{
    extern void dlaln2_(
        const int *ltrans, const int *na, const int *nw, const double *smin,
        const double *ca, const double *a, const int *lda, const double *d1,
        const double *d2, const double *b, const int *ldb, const double *wr,
        const double *wi, double *x, const int *ldx, double *scale,
        double *xnorm, int *info);

    int info;
    dlaln2_(&ltrans, &na, &nw, &smin, &ca, A, &ldA, &d1, &d2, B, &ldB, &wr, &wi,
        X, &ldX, scale, xnorm, &info);

    if (info == 1)
        printf("WARNING: DLALN2 had to perturb the entries\n");
}


void dgehrd(const int n, const int ilo, const int ihi,
    double *A, const int ldA, double *tau)
{
    extern void dgehrd_(
        const int *n, const int *ilo, const int *ihi, double *A, const int *ldA,
        double *tau, double *work, const int *lwork, int *info);

    int info;

    // Query optimal workspace size.
    int lwork = -1;
    double work_size = 0.0;
    dgehrd_(&n, &ilo, &ihi, A, &ldA, tau, &work_size, &lwork, &info);
    lwork = (int) work_size;

    double *work = malloc(lwork * sizeof(double));
    dgehrd_(&n, &ilo, &ihi, A, &ldA, tau, work, &lwork, &info);
    free(work);

    if (info != 0)
        printf("WARNING: dgehrd had illegal %d-th argument.\n", info);
}


void dhseqr(const char job, const char compz,
    const int n, const int ilo, const int ihi, double *H, const int ldH,
    double *wr, double *wi, double *Z, const int ldZ,
    double *work, const int lwork, int *info)
{
    extern void dhseqr_(const char *job, const char *compz, const int *n,
        const int *ilo, const int *ihi, double *H, const int *ldH,
        double *wr, double *im, double *Z, const int *ldZ,
        double *work, const int *lwork, int *info);

    dhseqr_(&job, &compz, &n, &ilo, &ihi, H, &ldH, wr, wi, Z, &ldZ, work,
        &lwork, info);
}
