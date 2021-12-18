#ifndef SOLVE_TASK_H
#define SOLVE_TASK_H

#include "utils.h"

/**
 * @brief Small robust backward substitution.
 *
 * @param[in] cos         n-by-m matrix storing the cosine component of the Givens
 *                        rotations computed during the reduction. The entry
 *                        cos[i + k * ldcos] stores the component
 *                        corresponding to shift k and Givens rotation i.
 * @param[in] ldcos       Leading dimension of cos.
 * @param[in] sin         Sine components.
 * @param[in] ldsin       Leading dimension of sin.
 * @param[in] n           Row count of matrix H.
 * @param[in] H           Matrix of size n-by-(n-1) such that [H, Rtildes(:,k)]
 *                        is upper Hessenberg.
 * @param[in] ldH         Leading dimension of H.
 * @param[in] hleft       The top column entry to the left of H.
 * @param[in,out] Hnorm   Upper bound of H (ignoring shifts). If negative on
 *                        entry, overwritten with the infinity norm of H.
 * @param[in] R           Crossover columns stored as n-by-m matrix. The k-th
 *                        column R[0:n-1 + k * ldR] is the crossover column
 *                        corresponding to wr[k].
 * @param[in] ldR         Leading dimension of R.
 * @param[in] nrhs        Number of eigenvalues and cross-over columns.
 * @param[in] wr          Vector of length m. Real eigenvalues.
 * @param[in] smlnum      Threshold when close eigenvalues shall be perturbed. 
 * @param[in,out] scales  Vector of m scaling factors.
 * @param[in,out] vrnorms Vector of length m. Upper bounds on columns of vr.
 * @param[in,out] vr      Matrix of size n-by-m. On entry, the right-hand sides.
 *                        On exit, the solution.
 * @param[in] ldvr        Leading dimension of vr.
 *
 * @returns 1 if the eigenvalues have been perturbed and 0 otherwise.
 *
 * See Algorithm 5, 9 in https://arxiv.org/pdf/2101.05063.pdf for details.
 */

int solve(
    const double *restrict cos, int ldcos, 
    const double *restrict sin, int ldsin,
    int n, const double *restrict H, int ldH, const double hleft,
    double *restrict Hnorm,
    const double *restrict R, int ldR,
    int nrhs, const double *restrict wr, const double smlnum,
    scaling_t *restrict scales,
    double *restrict vrnorms, double *restrict vr, int ldvr);


/**
 * @brief Factors a shifted Hessenberg matrix with Givens rotations into
 * an orthogonal Q and a triangular R, and solves the triangular system.
 *
 * @param[out] cos        On exit, n-by-m matrix storing the cosine component of
 *                        the Givens rotations computed during the reduction.
 *                        The entry cos[i + k * ldcos] stores the component
 *                        corresponding to shift k and Givens rotation i.
 *                        cos[0 + k * ldcos] is unused.
 * @param[out] ldcos      Leading dimension of cos.
 * @param[out] sin        Sine components.
 * @param[out] ldsin      Leading dimension of sin.
 *
 * See solve() for a description of the other paramters.
 * */
int factor_and_solve_R11(
    double *restrict cos, int ldcos, 
    double *restrict sin, int ldsin,
    int n, const double *restrict H, int ldH, double *restrict Hnorm,
    const double *restrict R, int ldR,
    int nrhs, const double *restrict wr, const double smlnum,
    scaling_t *restrict scales, 
    double *restrict vrnorms, double *restrict vr, int ldvr);

#endif
