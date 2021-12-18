#ifndef VALIDATION_H
#define VALIDATION_H


/**
 * @brief Validates the eigenvectors
 *
 * Verifies that each eigenvector satisfies
 * ||(H * y - s * y||_F / (||H||_F * ||y||_F + |s| * ||y||_F) < tol.
 * where y = Y(:,j) and s = wr[j].
 *
 * @param[in]  n        Dimension of the Hessenberg matrix H
 * @param[in]  H        Hessenberg matrix.
 * @param[in]  ldH      Leading dimension of H.
 * @param[in]  nrhs     Number of shifts/eigenvectors.
 * @param[in]  wr       Vector of length nrhs. Real eigenvalues.
 * @param[in]  Y        Eigenvector matrix of size n-by-nrhs. The i-th column
 *                      of Y is the corresponding eigenvector of wr[i].
 * @param[in]  ldY      Leading dimension of Y.
 * @param[in]  tol      Tolerance threshold.
 */
void validate_real_eigenvectors(
     int n, const double *restrict H, int ldH, int nrhs,
     const double *restrict wr, const double *restrict Y, int ldY,
     const double tol);


#endif
