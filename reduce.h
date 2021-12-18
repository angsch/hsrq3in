#ifndef REDUCE_H
#define REDUCE_H

#include "utils.h"

/** @brief Records Givens rotations and cross-over columns in a tiled reduction
 *         of a shifted Hessenberg matrix to triangular form.
 * 
 * @param[in]     H       Hessenberg matrix
 * @param[in]     ldH     Leading dimension of H
 * @param[in]     p       Partitioning of the eigenvector matrix.
 * @param[in]     wr      Real eigenvalues.
 * @param[out]    Rtildes Pointer to tiles of cross-over columns. On exit, the
 *                        cross-over columns stored in tile layout.
 * @param[out]    c
 * @param[out]    s       Pointer to tiles of Givens rotations. On exit, the
 *                        cosine and sine components of the Givens rotations.
 *
 * Details can be found in Algorithm 4, https://arxiv.org/pdf/2101.05063.pdf.
 * */
void tiledReduce(const double *restrict H, int ldH,
    partitioning_t *p, const double *restrict const wr,
    double *restrict *restrict Rtildes,
    double *restrict *restrict c, double *restrict *restrict s);


/**
 * @brief Computes the transpose of a Givens rotation.
 *
 * Finds c and s such that [  c  s ] * [ f ] = [ sqrt(f*f + g*g) ]
 *                         [ -s  c ]   [ g ]   [ 0               ]
 * @param[in]     f
 * @param[in]     g       The first and second component of the input vector.
 * @param[out]    c
 * @param[out]    s       The components of the Givens rotation.
 */
void givens(double f, double g, double *restrict c, double *restrict s);
#endif
