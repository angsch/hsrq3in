#include "reduce.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include <omp.h>


void givens(double f, double g, double *restrict c, double *restrict s)
{
    extern void dlartg_(const double *f, const double *g,
                        double *c,double *s, double *r);

    double r;
    dlartg_(&f, &g, c, s, &r);

    // Transpose.
    *s = (-1.0) * (*s);
}


static void reduceDiag(
    int n, const double *restrict H, int ldH, const double hleft,
    int nrhs, const double *restrict wr,
    const double *restrict const R, int ldR,
    double *restrict cos, int ldcos, double *restrict sin, int ldsin)
{
#define H(i,j) H[(i) + (j) * (size_t)ldH]
#define R(i,k) R[(i) + (size_t)ldR * (k)]
#define c(k,rhs) cos[(k) + (rhs) * (size_t)ldcos]
#define s(k,rhs) sin[(k) + (rhs) * (size_t)ldsin]

    // Running column.
    double r[n];

    for (int rhs = 0; rhs < nrhs; rhs++) {
        // The rightmost column is the cross-over column.
        memcpy(r, &R(0, rhs), n * sizeof(double));

        for (int k = n-1; k >= 1; k--) {
            givens(r[k], H(k,k-1), &c(k,rhs), &s(k,rhs)); // Generate G^T.
            for (int i = 0; i < k-1; i++) {
                r[i] = H(i,k-1) * c(k,rhs) + r[i] * s(k,rhs);
            }
            r[k-1] = (H(k-1,k-1) - wr[rhs]) * c(k,rhs) + r[k-1] * s(k,rhs);
        } // for k

        if (hleft != 0) {
            // Apply the final Givens rotations.
            givens(r[0], hleft, &c(0,rhs), &s(0,rhs));
        }
        else {
            // Identity matrix (padding).
            c(0,rhs) = 1.0;
            s(0, rhs) = 0.0;
        }
    } // for rhs

#undef c
#undef s
#undef H
#undef R
}



// Loop-blocked version of reduceOffdiag().
static void reduceOffdiagBlocked(
    int k, int n, const double *restrict H, int ldH,
    int nrhs, const double *restrict wr, int applyShift,
    const double *restrict R_right, double *restrict R_left, int ldR,
    double *restrict cos, int ldcos, double *restrict sin, int ldsin)
{
    // Aliases.
    int nrows = k;
    int ncols = n;

    // The driver partitions the matrix such that all calls to this routine
    // exhibit a column count that is an integer multiple of 4.
    //
    // If this routine is used with a different partitioning, a generalisation
    // of Algorithm 4 in C BEATTIE, Z DRMAČ, S GUGERCIN: A note of shifted
    // Hessenberg systems and frequency response computation, ACM Toms, Vol. 38,
    // No. 2, Article 12 (2011) is necessary.
    assert(ncols % 4 == 0);

#define H(i,j) H[(i) + (j) * (size_t)ldH]
#define R_right(i,k) R_right[(i) + (size_t)ldR * (k)]
#define R_left(i,k) R_left[(i) + (size_t)ldR * (k)]
#define c(k,rhs) cos[(k) + (rhs) * (size_t)ldcos]
#define s(k,rhs) sin[(k) + (rhs) * (size_t)ldsin]

    // Running column.
    double r[nrows];

    for (int rhs = 0; rhs < nrhs; rhs++) {
        // The rightmost column is the cross-over column.
        memcpy(r, &R_right(0, rhs), nrows * sizeof(double));

        for (int k = n-1; k >= 4; k-=4) {
            // Apply four Givens rotations at once.
            for (int i = 0; i < nrows; i++) {
                r[i] = c(k-3,rhs)                                      * H(i,k-3)
                     + s(k-3,rhs) * c(k-2,rhs)                         * H(i,k-2)
                     + s(k-3,rhs) * s(k-2,rhs) * c(k-1,rhs)            * H(i,k-1)
                     + s(k-3,rhs) * s(k-2,rhs) * s(k-1,rhs) * c(k,rhs) * H(i,k)
                     + s(k-3,rhs) * s(k-2,rhs) * s(k-1,rhs) * s(k,rhs) * r[i];
            }
        }


        // Apply the last four rotations, compute the cross-over column.
        {
            // Rows 0, ..., nrows-2.
            for (int i = 0; i < nrows-1; i++) {
                R_left(i,rhs) = c(0,rhs)                         * H(i,0)
                     + s(0,rhs) * c(1,rhs)                       * H(i,1)
                     + s(0,rhs) * s(1,rhs) * c(2,rhs)            * H(i,2)
                     + s(0,rhs) * s(1,rhs) * s(2,rhs) * c(3,rhs) * H(i,3)
                     + s(0,rhs) * s(1,rhs) * s(2,rhs) * s(3,rhs) * r[i];
            }

            // Row n-1.
            R_left(nrows-1,rhs) =
                       c(0,rhs) * (H[nrows-1] - applyShift * wr[rhs])
                     + s(0,rhs) * c(1,rhs)                       * H(nrows-1,1)
                     + s(0,rhs) * s(1,rhs) * c(2,rhs)            * H(nrows-1,2)
                     + s(0,rhs) * s(1,rhs) * s(2,rhs) * c(3,rhs) * H(nrows-1,3)
                     + s(0,rhs) * s(1,rhs) * s(2,rhs) * s(3,rhs) * r[nrows-1];
        }

    } // for rhs
#undef c
#undef s
#undef R_right
#undef R_left
#undef H
}


static void reduceOffdiag(
    int k, int n, const double *restrict H, int ldH,
    int nrhs, const double *restrict wr, int applyShift,
    const double *restrict R_right, double *restrict R_left, int ldR,
    double *restrict cos, int ldcos, double *restrict sin, int ldsin)
{
#define H(i,j) H[(i) + (j) * (size_t)ldH]
#define R_right(i,k) R_right[(i) + (size_t)ldR * (k)]
#define R_left(i,k) R_left[(i) + (size_t)ldR * (k)]
#define c(k,rhs) cos[(k) + (rhs) * (size_t)ldcos]
#define s(k,rhs) sin[(k) + (rhs) * (size_t)ldsin]

    // Aliases.
    int nrows = k;
    int ncols = n;

    // Running column.
    double r[nrows];

    for (int rhs = 0; rhs < nrhs; rhs++) {
        // The rightmost column is the cross-over column.
        memcpy(r, &R_right(0, rhs), nrows * sizeof(double));

        for (int k = ncols-1; k >= 1; k--) {
            // Apply G^T = [ c -s ]
            //             [ s  c ]
            // R(1:k,k-1) = [ H(1:k,k-1) - mu * E(1:k,k-1), R(1:k,k) ] * [ c
            //                                                             s ];
            for (int i = 0; i < nrows; i++) {
                r[i] = H(i,k) * c(k,rhs) + r[i] * s(k,rhs);
            }
        }

        // Compute the cross-over column.
        {
            // Rows 0, ..., nrows-2.
            for (int i = 0; i < nrows-1; i++) {
                R_left(i,rhs) = H[i] * c(0,rhs) + r[i] * s(0,rhs);
            }

            // Row n-1.
            R_left(nrows-1,rhs) =
                            (H[nrows-1] - applyShift * wr[rhs]) * c(0,rhs)
                          + r[nrows-1] * s(0,rhs);
        }
    } // for rhs

#undef c
#undef s
#undef R_right
#undef R_left
#undef H
}


void tiledReduce(const double *restrict H, int ldH,
    partitioning_t *p, const double *restrict const wr,
    double *restrict *restrict Rtildes,
    double *restrict *restrict c, double *restrict *restrict s)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int num_rhs_tiles = p->num_tile_cols;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    int n = first_row[num_tiles];

#define H(i,j) H[(i) + (j) * (size_t)ldH]

    // Unpack workspaces in tile layout.
#define Rtildes(tli,tlk,tlj) Rtildes[(tli) + num_tiles * (tlk) + (size_t)num_tiles * num_rhs_tiles * (tlj)]
#define c(tli,tlk) c[(tli) + num_tiles * (tlk)]
#define s(tli,tlk) s[(tli) + num_tiles * (tlk)]

    #pragma omp parallel shared(c) shared(s) shared(Rtildes)
    #pragma omp single nowait
    for (int tlk = 0; tlk < num_rhs_tiles; tlk++) {
        // The rightmost tile columns require initialisation of the initial
        // right cross-over columns with H(:,n) - wr * I(:,n).

        // Process Rnn.
        #pragma omp task depend(out: c(num_tiles - 1,tlk))
        {
            // Dimensions of the right-hand side tile.
            int nrows = first_row[num_tiles] - first_row[num_tiles - 1];
            int nrhs = first_col[tlk + 1] - first_col[tlk];

            // Locate data.
            int l = first_row[num_tiles - 1];
            double hleft = H(l,l-1);
            int k = first_col[tlk];

            // Preprocessing: Set initial cross-over columns.
            {
                const double *restrict Hin = &H(l, n - 1);
                double *restrict r = Rtildes(num_tiles - 1, tlk, num_tiles - 1);
                for (int rhs = 0; rhs < nrhs; rhs++) {
                    memcpy(r + rhs * nrows, Hin, nrows * sizeof(double));
                    r[rhs * nrows + nrows - 1] -= wr[k + rhs];
                }
            }

            reduceDiag(nrows, &H(l,l), ldH, hleft, nrhs, wr + k,
                Rtildes(num_tiles - 1, tlk, num_tiles - 1), nrows,
                c(num_tiles - 1,tlk), nrows, s(num_tiles - 1,tlk), nrows);
        } // task Rnn
        for (int tli = num_tiles - 2; tli >= 0; tli--) {
            // Process superdiagonal tiles Rin.
            #pragma omp task \
                depend(in: c(num_tiles - 1,tlk)) \
                depend(out: Rtildes(tli,tlk,num_tiles-2))
            {
                // Compute dimensions.
                int nrows = first_row[tli + 1] - first_row[tli];
                int ncols = first_row[num_tiles] - first_row[num_tiles-1];
                int nrhs = first_col[tlk + 1] - first_col[tlk];

                // Preprocessing: Set initial cross-over columns.
                {
                    const double *restrict Hin = &H(first_row[tli], n-1);
                    double *restrict r = Rtildes(tli, tlk, num_tiles - 1);
                    for (int rhs = 0; rhs < nrhs; rhs++) {
                        memcpy(r + rhs * nrows, Hin, nrows * sizeof(double));
                    }
                }

                // Are we just above the diagonal?
                int applyShift = 0;
                if (tli == num_tiles - 2)
                    applyShift = 1;

                // Leading dimensions.
                int ldR = nrows;
                int ldc = ncols;
                int lds = ncols;

                // Locate data.
                int k = first_col[tlk];
                const double *restrict Hin = 
                    &H(first_row[tli], first_row[num_tiles-1] - 1); // H(I, J^-)

                reduceOffdiagBlocked(nrows, ncols, Hin, ldH,
                    nrhs, wr + k, applyShift,
                    Rtildes(tli, tlk, num_tiles - 1),
                    Rtildes(tli, tlk, num_tiles - 2), ldR,
                    c(num_tiles - 1,tlk), ldc, s(num_tiles - 1,tlk), lds);
            } // task Rin
        }

        // Center tile columns.
        for (int tlj = num_tiles - 2; tlj >= 1; tlj--) {
            // Process diagonal Rjj.
            #pragma omp task \
              depend(in: Rtildes(tlj,tlk,tlj)) \
              depend(out: c(tlj,tlk))
            {
                // Compute the dimensions.
                int nrows = first_row[tlj + 1] - first_row[tlj];
                int nrhs = first_col[tlk + 1] - first_col[tlk];

                // Locate data.
                int l = first_row[tlj];
                double hleft = H(l,l-1);
                int k = first_col[tlk];

                reduceDiag(nrows, &H(l,l), ldH, hleft, nrhs, wr + k,
                    Rtildes(tlj,tlk,tlj), nrows,
                    c(tlj,tlk), nrows, s(tlj,tlk), nrows);
            } // task Rjj

            for (int tli = tlj - 1; tli >= 0; tli--) {
                // Process superdiagonal tiles Rij.
                #pragma omp task                       \
                  depend(in: c(tlj,tlk))               \
                  depend(in: Rtildes(tli,tlk,tlj))     \
                  depend(out: Rtildes(tli,tlk,tlj-1))
                {
                    // Compute the dimensions.
                    int nrows = first_row[tli + 1] - first_row[tli];
                    int ncols = first_row[tlj + 1] - first_row[tlj];
                    int nrhs = first_col[tlk + 1] - first_col[tlk];

                    // Are we just above the diagonal?
                    int applyShift = 0;
                    if (tli == tlj - 1)
                        applyShift = 1;

                    // Leading dimensions.
                    int ldR = nrows;
                    int ldc = ncols;
                    int lds = ncols;

                    // Locate data.
                    int k = first_col[tlk];
                    const double *restrict Hij = 
                        &H(first_row[tli],first_row[tlj] - 1); // H(I, J^-)

                    reduceOffdiagBlocked(nrows, ncols,
                        Hij, ldH, nrhs, wr + k, applyShift,
                        Rtildes(tli,tlk,tlj), // in
                        Rtildes(tli,tlk,tlj-1), ldR, // out
                        c(tlj,tlk), ldc, s(tlj,tlk), lds);
                } // task Rij
            } // for tli
        } // for tlj

        // Leftmost tile column.
        // No computation - cos, sin are computed in factor_and_solve_R11.

    } // for tlk
} // parallel
