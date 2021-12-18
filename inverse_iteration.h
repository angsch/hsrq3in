#ifndef INVERSE_ITERATION_H_
#define INVERSE_ITERATION_H_

/** @brief Computes eigenvectors corresponding to real eigenvalues by inverse iteration.
 * 
 * @param[in]  n        Dimension of the Hessenberg matrix H. n > 0.
 * @param[in]  H        Hessenberg matrix.
 * @param[in]  ldH      Leading dimension of H. ldH >= n.
 * @param[in]  nrhs     Number of shifts/eigenvectors.
 * @param[in,out]  wr   Vector of length nrhs. Real eigenvalues. May be reordered.
 * @param[in]  tlsz     Tile size used to partition the rows of vr. Must be a positive
 *                      integer multiple of 4.
 * @param[in]  rhs_tlsz Tile size used to partition the columns of vr. rhs_tlsz > 0.
 * @param[out] vr       Eigenvector matrix of size n x nrhs. The k-th column
 *                      stores an eigenvector corresponding to wr[k].
 * @param[in]  ldvr     Leading dimension of vr.
 * 
 * */
void tiled_inverse_iteration(
    int n, const double *restrict H, int ldH,
    int nrhs, double *restrict wr,
    int tlsz, int rhs_tlsz, double *restrict vr, int ldvr);

#endif
