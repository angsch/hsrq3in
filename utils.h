#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef struct {
    int num_tile_rows; // num_tile_rows + 1 exists
    int num_tile_cols; // num_tile_cols + 1 exists
    int *first_row;
    int *first_col;
} partitioning_t;

#ifdef INTSCALING
typedef int scaling_t;
#else
typedef double scaling_t;
#endif

#define NO_RESCALE 0
#define RESCALE 1

double get_time (void);

int get_size_with_padding(const int n);


static inline void set_zero(int n, int m, double *restrict const A, int ldA)
{
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            A[(size_t) j * ldA + i] = 0.0;

}

static inline int min(int a, int b)
{
    return a < b ? a : b;
}

static inline int max(int a, int b)
{
    return a > b ? a : b;
}

void scale_tile(int m, int n, 
    double *restrict const X, int ldX, const scaling_t *beta);

void scale_excluding(int m, int n, int ilo, int ihi, int jlo, int jhi,
    double *restrict const X, int ldX, const scaling_t *beta);

void scale_except(
    int n, int skip, double *restrict const x, const scaling_t * beta);


void dgemm(
    const char transa, const char transb,
    const int m, const int n, const int k,
    const double alpha, const double *restrict const A, const int ldA,
    const double *restrict const B, const int ldB,
    const double beta, double *restrict C, const int ldC);


void dgemv(
    const char trans,
    const int m, const int n,
    const double alpha, const double *restrict const A, const int ldA,
    const double *restrict x, int incx,
    const double beta, double *restrict const y, int incy);


inline double vector_infnorm(int n, const double *x)
{
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double abs =  fabs(x[i]);
        if (abs > norm) {
            norm = abs;
        }
    }
    return norm;
}


double dlange(const char norm, const int m, const int n,
    const double *restrict const A, int ldA);


void dlaln2(int ltrans, int na, int nw, double smin, double ca, double *A,
    int ldA, double d1, double d2, double *B, int ldB, double wr, double wi,
    double *X, int ldX, double *scale, double *xnorm);


void dgehrd(const int n, const int ilo, const int ihi,
    double *A, const int ldA, double *tau);


void dhseqr(const char job, const char compz,
    const int n, const int ilo, const int ihi, double *H, const int ldH,
    double *wr, double *wi, double *Z, const int ldZ,
    double *work, const int lwork, int *info);


#endif
