#include "validation.h"
#include "utils.h"
#include "partition.h"

#include <mm_malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>


// Compute || X_re ||^2 where || . || is the Frobenius norm.
static double vector_frobeniusnorm2(int n, const double *restrict const X)
{
    double sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum2 += X[i] * X[i];
    }

    return sum2;
}

static double hessenberg_frobeniusnorm(int n, const double *restrict H, int ldH)
{
    double norm = 0.0;
    double *sum2 = (double *) malloc(n * sizeof(double));

    #pragma omp parallel shared(sum2)
    {
        #pragma omp for
        for (int j = 0; j < n; j++) {
            const double *restrict Hj = H + j * ldH;
            int len = min(j+2, n);
            sum2[j] = vector_frobeniusnorm2(len, Hj);
        }

        #pragma omp parallel for reduction(+:norm)
        for (int j = 0; j < n; j++)
            norm += sum2[j];
    }

    free(sum2);
    return sqrt(norm);
}


// C := beta * C + alpha * A * B, where A is upper Hessenberg.
static void matmul_Hessenberg(
    double alpha,
    double ***A_tiles, int ldA, partitioning_t *part_A,
    double ***B_tiles, int ldB, partitioning_t *part_B,
    double beta,
    double ***C_tiles, int ldC, partitioning_t *part_C)
{
    #pragma omp parallel
    #pragma omp single
     for (int j = 0; j < part_C->num_tile_cols; j++) {
         for (int i = 0; i < part_C->num_tile_rows; i++) {
            // Compute C(i,j) := SUM [A(i,l) * B(l,j)]
            #pragma omp task
            {
                // Compute the actual number of rows and column.
                const int m = part_C->first_row[i + 1] - part_C->first_row[i];
                const int n = part_C->first_col[j + 1] - part_C->first_col[j];

                // C(i,j) := beta * C(i,j) + alpha * A(i,l) * B(l,j).

                // Compute inner dimension = cols of A/rows of B.
                const int k = part_A->first_col[part_A->num_tile_cols];

                dgemm('N', 'N', m, n, k,
                      alpha, A_tiles[i][0], ldA, B_tiles[0][j], ldB,
                      beta, C_tiles[i][j], ldC);
            }
        }
    }
}

void validate_real_eigenvectors(
     int n, const double *restrict H, int ldH, int nrhs,
     const double *restrict wr, const double *restrict Y, int ldY,
     const double tol)
{
    // Compute ||H||_F.
    double normH = hessenberg_frobeniusnorm(n, H, ldH);

    // Allocate workspace.
    double *work = (double *) calloc(ldY * nrhs, sizeof(double));

    // work := H * Y.
    {
        // Create a temporary partitioning.
        int tlsz = 400;
        int num_tile_rows = (n + tlsz - 1) / tlsz;
        int num_tile_cols = (nrhs + tlsz - 1) / tlsz;
        int *first_row = (int *) malloc((num_tile_rows + 1) * sizeof(int));
        int *first_col = (int *) malloc((num_tile_cols + 1) * sizeof(int));
        partition(n, num_tile_rows, tlsz, first_row);
        partition(nrhs, num_tile_cols, tlsz, first_col);
        partitioning_t part_H = {.num_tile_rows = num_tile_rows,
                                 .num_tile_cols = num_tile_rows,
                                 .first_row = first_row,
                                 .first_col = first_row};
        partitioning_t part_Y = {.num_tile_rows = num_tile_rows,
                                 .num_tile_cols = num_tile_cols,
                                 .first_row = first_row,
                                 .first_col = first_col};
        double ***H_tiles = malloc(num_tile_rows * sizeof(double **));
        for (int i = 0; i < num_tile_rows; i++) {
            H_tiles[i] = malloc(num_tile_rows * sizeof(double *));
        }
        partition_matrix(H, ldH, &part_H, H_tiles);

        double ***Y_tiles = malloc(num_tile_rows * sizeof(double **));
        for (int i = 0; i < num_tile_rows; i++) {
            Y_tiles[i] = malloc(num_tile_cols * sizeof(double *));
        }
        partition_matrix(Y, ldY, &part_Y, Y_tiles);

        double ***work_tiles = malloc(num_tile_rows * sizeof(double **));
        for (int i = 0; i < num_tile_rows; i++) {
            work_tiles[i] = malloc(num_tile_cols * sizeof(double *));
        }
        partition_matrix(work, ldY, &part_Y, work_tiles);

        // work := H * Y
        matmul_Hessenberg(1.0, H_tiles, ldH, &part_H, Y_tiles, ldY, &part_Y,
                          0.0, work_tiles, ldY, &part_Y);

        // Clean up.
        free(first_row);
        free(first_col);
        for (int i = 0; i < num_tile_rows; i++) {
            free(H_tiles[i]);
            free(Y_tiles[i]);
            free(work_tiles[i]);
        }
        free(H_tiles);
        free(Y_tiles);
        free(work_tiles);
    }

    // work := work - Y * diag(wr) and compute relative error.
    {
        for (int k = 0; k < nrhs; k++) {
            // Locate k-th column.
            const double *restrict Yk = Y + k * ldY;
            double *restrict r = work + k * ldY;

            // Compute the residual.
            for (int i = 0; i < n; i++)
                r[i] = r[i] - wr[k] * Yk[i];

            // Norm of the eigenvector.
            double normY = dlange('F', n, 1, Yk, n);

            // Norm of the residual.
            double normR = dlange('F', n, 1, r, n);

            const double abslambda = fabs(wr[k]);
            double err = normR / (normH * normY + abslambda * normY);

            // Check if result exceeds tolerance or is NaN.
            if (err > tol || err != err)
                printf("Eigenvector stored in column %d has relative error %.6e.\n", k, err);
        }
    }

    // Clean up.
    _mm_free(work);
}
