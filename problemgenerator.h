#ifndef PROBLEMGENERATOR_H
#define PROBLEMGENERATOR_H


/**
 * @brief Generates a Hessenberg matrix with well-separated real eigenvalues.
 *
 * @param[in]  n              The dimension of the Hessenberg matrix H.
 * @param[out] H              On exit, the Hessenberg matrix.
 * @param[in]  ldH            The leading dimension of H.
 * @param[out] wr             Real parts of the eigenvalues.
 * @param[out] wi             Imaginary parts of the eigenvalues.
 *
 * See the first test problem in https://arxiv.org/pdf/2101.05063.pdf,
 * Section 5.2.
 * */
void generate_hessenberg_with_separated_eigenvalues(
    int n, double *restrict H, int ldH,
    double *restrict wr, double *restrict wi);


#endif 
