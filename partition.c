#include "utils.h"
#include "partition.h"

#include <assert.h>
#include <stdio.h>

void partition(
    int n, int num_tiles, int tlsz, int *restrict p)
{
    // Partition into tiles of size tlsz. The first tile may be smaller.
    for (int i = 1; i < num_tiles; i++)
        p[i] = n - (num_tiles - i) * tlsz;

    // Pad so that the size computes p[i + 1] - p[i].
    p[0] = 0;
    p[num_tiles] = n;
}


void partition_matrix(
    const double *restrict A, int ldA,
    const partitioning_t *restrict p,
    double ***restrict A_tiles)
{
    // Extract row and column partitioning.
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    const int num_tile_rows = p->num_tile_rows;
    const int num_tile_cols = p->num_tile_cols;

    #define A(i,j) A[(i) + (j) * ldA]
    for (int i = 0; i < num_tile_rows; i++) {
        for (int j = 0; j < num_tile_cols; j++) {
            A_tiles[i][j] = &A(first_row[i], first_col[j]);
        }
    }
    #undef A

}
