#include "inverse_iteration.h"
#include "partition.h"
#include "problemgenerator.h"
#include "utils.h"
#include "validation.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>


int main(int argc, char **argv)
{
    // Dimension of the system matrix, tile size.
    int n, tlsz, rhs_tlsz;

    // Problem generation parameters.
    double select_frac;

    // Seed for random number generator.
    unsigned int seed;

    if (argc != 6) {
        printf("Usage %s n tlsz rhs-tlsz select-frac seed\n", argv[0]);
        printf("n:             Dimension of square matrix H\n");
        printf("tlsz:          Tile size\n");
        printf("rhs-tlsz:      Tile size for column partitioning of the right-hand sides\n");
        printf("select-frac:   Fraction of the selected eigenvalues\n");
        printf("seed:          Seed for the random number generator\n");
        return EXIT_FAILURE;
    }

    // Set inputs.
    n = atoi(argv[1]);
    tlsz = atoi(argv[2]);
    rhs_tlsz = atoi(argv[3]);
    select_frac = atof(argv[4]);
    seed = (unsigned int)atoi(argv[5]);
    seed = 0;

    // Initialize the random number generator.
    srand(seed);

    // Validate the inputs.
    assert(n > 0);
    assert(tlsz > 0);
    assert(rhs_tlsz > 0);
    assert(select_frac >= 0.0);
    assert(select_frac <= 1.0);

    // Process tile sizes.
    if (tlsz % 4 != 0) {
        tlsz = ((tlsz + 3)/4 ) * 4;
        printf("INFO: tlsz must be divisible by 4. Set tlsz to %d.\n", tlsz);
    }

    // Print configuration.
    printf("Configuration:\n");
    #pragma omp parallel
    #pragma omp single
    printf("  OpenMP threads = %d\n", omp_get_num_threads());
    printf("  n = %d\n", n);
    printf("  tlsz = %d\n", tlsz);
    printf("  rhs-tlsz = %d\n", rhs_tlsz);
    printf("  select-frac = %.2f\n", select_frac);
    printf("  Seed = %u\n", seed);

    // Number of eigenvectors and leading dimensions.
    int nrhs = n * select_frac;
    int ldH = get_size_with_padding(n);
    int ldY = get_size_with_padding(n);

    // Tolerance threshold.
    const double tol = 1.e-12;

    // Allocate matrices and workspaces.
    double *restrict H;
    double *restrict Y;
    H = (double *) _mm_malloc((size_t)ldH * n * sizeof(double), ALIGNMENT);
    double *wr = (double *) _mm_malloc((size_t)n * sizeof(double), ALIGNMENT);
    double *wi = (double *) _mm_malloc((size_t)n * sizeof(double), ALIGNMENT);
    Y = (double *) _mm_malloc((size_t)ldY * nrhs * sizeof(double), ALIGNMENT);

    memset(Y, 0.0, ldY * nrhs * sizeof(double));

    printf("Setting up the problem...\n");
    generate_hessenberg_with_separated_eigenvalues(n, H, ldH, wr, wi);

    // So far only real eigenvalues are supported.
    for (int i = 0; i < n; i++)
        assert(wi[i] == 0.0);

    double tm_start, tm_end;
    printf("Solve...\n");
    tm_start = get_time();
    tiled_inverse_iteration(n, H, ldH, nrhs, wr, tlsz, rhs_tlsz, Y, ldY);
    tm_end = get_time();
    printf("Execution time [s]= %.2f\n", tm_end - tm_start);

    printf("Validate...\n");
    validate_real_eigenvectors(n, H, ldH, nrhs, wr, Y, ldY, tol);

    // Clean up.
    _mm_free(H);
    _mm_free(Y);
    _mm_free(wr);
    _mm_free(wi);

    return EXIT_SUCCESS;
}
