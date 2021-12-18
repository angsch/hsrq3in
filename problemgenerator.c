#include "problemgenerator.h"
#include "partition.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include <omp.h>


static double random_double (double low, double high)
{
    double x = (double) rand () / RAND_MAX;
    return low + x * (high - low);
}


static void generate_2x2_block(double *restrict const T, const int ldT,
    const double lambda_re, const double lambda_im)
{
#define T(i,j) T[(i) + (size_t)ldT * (j)]

    T(0,0) =  lambda_re;  T(0,1) = lambda_im;
    T(1,0) = -lambda_im;  T(1,1) = lambda_re;

#undef T
}

static void generate_quasitriangular_tile(
    int m, double *T, int ldT,
    const double *restrict const wr, const double *restrict const wi)
{
#define T(i,j) T[(i) + (size_t)ldT * (j)]

    // Zero out lower triangular part.
    for (int i = 0; i < m; i++)
        for (int j = 0; j < i; j++)
            T(i,j) = 0.0;

    // Superdiagonal.
    for (int j = 0; j < m; j++)
        for (int i = 0; i < j; i++)
            T(i,j) = random_double(0.0, 1.0);

    // Set diagonal.
    for (int i = 0; i < m; i++) {
        if (wi[i] == 0.0) {
            // Real eigenvalue.
            T(i,i) = wr[i];
        }
        else {
            // Pair of complex conjugate eigenvalues.
            generate_2x2_block(&T(i,i), ldT, wr[i], wr[i]);

            // Skip the next eigenvalue.
            i++;
        }
    }

#undef T
}

static void generate_dense_tile(
    int m, int n, double *A, int ldA)
{
#define A(i,j) A[(i) + (size_t)ldA * (j)]

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A(i,j) = random_double(0.0, 1.0);

#undef A
}


static void generate_tiled_quasitriangular_matrix(
    double ***T_tiles, partitioning_t *p, int ldT,
    const double *restrict wr, const double *restrict wi)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
#ifndef NDEBUG
    const int m = first_row[num_tiles];
    const int n = first_col[num_tiles];
#endif

    // We expect a square matrix.
    assert(m == n);
    assert(p->num_tile_rows == p->num_tile_cols);

    #pragma omp parallel
    #pragma omp single nowait
    {
        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                if (i == j) {
                    #pragma omp task
                    {
                        int num_rows = first_row[i + 1] - first_row[i];

                        generate_quasitriangular_tile(
                            num_rows, T_tiles[i][i], ldT,
                            wr + first_row[i], wi + first_row[i]);
                    }
                }
                else if (j > i) {
                    #pragma omp task
                    {
                        int num_rows = first_row[i + 1] - first_row[i];
                        int num_cols = first_col[j + 1] - first_col[j];

                        generate_dense_tile(
                            num_rows, num_cols, T_tiles[i][j], ldT);
                    }
                }
                else { // Lower triangular part.
                    #pragma omp task
                    {
                        int num_rows = first_row[i + 1] - first_row[i];
                        int num_cols = first_col[j + 1] - first_col[j];

                        // Set tile to zero.
                        set_zero(num_rows, num_cols, T_tiles[i][j], ldT);
                    }
                }
            } // for j
        } // for i

    } // parallel region
}


static void generate_diagonal_householder_tile(
    int n, int ld, double *const H, const double *restrict const v)
{
#define H(i,j) H[(i) + (j) * ld]

    for(int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i == j) {
                H(i,j) = 1.0 - 2.0 * v[i] * v[j];
            }
            else {
                H(i,j) = -2.0 * v[i] * v[j];
            }
        }
    }

#undef H
}


static void generate_offdiagonal_householder_tile(
    int n, int m, int ld, double *const Hij, const double *restrict const vi,
    const double *restrict const vj)
{
#define Hij(i,j) Hij[(i) + (j) * (size_t)ld]

    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            Hij(i,j) = -2.0 * vi[i] * vj[j];

#undef Hij
}



static void generate_tiled_householder_matrix(
    double ***restrict Q_tiles, partitioning_t *p, int ldQ)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
#ifndef NDEBUG
    const int m = first_row[num_tiles];
#endif
    const int n = first_col[num_tiles];

    // We expect a square matrix.
    assert(m == n);
    assert(p->num_tile_rows == p->num_tile_cols);

    // Generate random unit vector v.
    double *v = (double *) _mm_malloc(n * sizeof(double), ALIGNMENT);
    for (int i = 0; i < n; i++)
        v[i] = 2.0 * (1.0 * rand() / RAND_MAX) - 1.0;
    extern double dnrm2_(const int *n, const double *x, const int *incx);
    const int incx = 1;
    double norm = dnrm2_(&n, v, &incx);
    for (int i = 0; i < n; i++)
        v[i] = v[i] / norm;

    // Compute Q := I - 2 * v * v^T.
    #pragma omp parallel
    #pragma omp single
    for (int i = 0; i < num_tiles; i++) {
        for (int j = 0; j < num_tiles; j++) {
            if (i == j) {
                // Compute Householder tile Qii := I - 2 * vi * vi^T.
                #pragma omp task
                {
                    const int num_rows = first_row[i + 1] - first_row[i];

                    generate_diagonal_householder_tile(
                        num_rows, ldQ, Q_tiles[i][i], v + first_row[i]);
                }
            }
            else {
                // Compute Householder tile Qij := -2 * vi * vj^T.
                #pragma omp task
                {
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    generate_offdiagonal_householder_tile(
                        num_rows, num_cols, ldQ, Q_tiles[i][j],
                        v + first_row[i], v + first_col[j]);
                }
            }
        }
    }

    _mm_free(v);
}

void similarity_transform(double ***H_tiles, int ldH,
    double ***Q_tiles, int ldQ, partitioning_t *p)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    const int n = first_col[num_tiles];

    // Allocate workspace.
    int ldC = ldH;
    double *C = (double *) _mm_malloc(n * ldC * sizeof(double), ALIGNMENT);
    double ***C_tiles = malloc(num_tiles * sizeof(double **));
    for (int i = 0; i < num_tiles; i++) {
        C_tiles[i] = malloc(num_tiles * sizeof(double *));
    }
    partition_matrix(C, ldC, p, C_tiles);

    // work := Q * H.
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                // Compute C(i,j) := SUM [Q(i,l) * H(l,j)]
                //                   l=0
                #pragma omp task
                {
                    // Compute the dimensions of C(i,j).
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    dgemm('N', 'N', num_rows, num_cols, first_col[num_tiles],
                        1.0, Q_tiles[i][0], ldQ,
                        H_tiles[0][j], ldH,
                        0.0, C_tiles[i][j], ldC);
                }
            }
        }
    }

    // H := C * Q^T.
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < num_tiles; i++) {
            for (int j = 0; j < num_tiles; j++) {
                // Compute H(i,j) := SUM [C(i,l) * Q(l,j)^T]
                //                   l=0
                #pragma omp task
                {
                    // Compute the dimensions of C(i,j).
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    dgemm('N', 'T', num_rows, num_cols, first_col[num_tiles],
                        1.0, C_tiles[i][0], ldC,
                        Q_tiles[j][0], ldQ,
                        0.0, H_tiles[i][j], ldH);
                }
            }
        }
    }

    // Clean up.
    for (int i = 0; i < num_tiles; i++) {
        free(C_tiles[i]);
    }
    free(C_tiles);
    _mm_free(C);
}

void generate_hessenberg_with_separated_eigenvalues(
    int n, double *restrict H, int ld,
    double *restrict wr, double *restrict wi)
{
    // Pick well-separated real eigenvalues.
    for (int j = 0; j < n; j++) {
        wr[j] = j + 1.0;
        wi[j] = 0.0;
    }

    // Generate partitionings for task-parallel initialization.
    int tlsz = 100;
    int num_tiles = (n + tlsz - 1) / tlsz;
    int *first_row = (int *) malloc((num_tiles + 1) * sizeof(int));
    int *first_col = first_row;
    partition(n, num_tiles, tlsz, first_row);
    partitioning_t p = {.num_tile_rows = num_tiles,
                        .num_tile_cols = num_tiles,
                        .first_row = first_row,
                        .first_col = first_col};
    double ***H_tiles = malloc(num_tiles * sizeof(double **));
    for (int i = 0; i < num_tiles; i++) {
        H_tiles[i] = malloc(num_tiles * sizeof(double *));
    }
    partition_matrix(H, ld, &p, H_tiles);
    double *Q = (double *) _mm_malloc(n * ld * sizeof(double), ALIGNMENT);
    double ***Q_tiles = malloc(num_tiles * sizeof(double **));
    for (int i = 0; i < num_tiles; i++) {
        Q_tiles[i] = malloc(num_tiles * sizeof(double *));
    }
    partition_matrix(Q, ld, &p, Q_tiles);

    // Allocate workspace.
    double *tau = (double *) _mm_malloc((size_t)n * sizeof(double), ALIGNMENT);

    // Generate Hessenberg matrix.
    generate_tiled_quasitriangular_matrix(H_tiles, &p, ld, wr, wi);
    generate_tiled_householder_matrix(Q_tiles, &p, ld);
    similarity_transform(H_tiles, ld, Q_tiles, ld, &p); // H := Q * H * Q^T
    dgehrd(n, 1, n, H, ld, tau); // Hessenberg reduction.

    // Zero out the lower half.
    #pragma omp parallel for
    for (int j = 0; j < n; j++)
        for (int i = j + 2; i < n; i++)
            H[i + (size_t)ld * j] = 0.0;

    // Clean up.
    free(first_row);
    for (int tli = 0; tli < num_tiles; tli++) {
        free(Q_tiles[tli]);
        free(H_tiles[tli]);
    }
    free(H_tiles);
    free(Q_tiles);
    _mm_free(Q);
    _mm_free(tau);
}
