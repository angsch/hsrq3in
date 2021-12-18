#ifndef SHIFTED_HESSENBERG_SOLVE_H_
#define SHIFTED_HESSENBERG_SOLVE_H_

#include "utils.h"

/** @brief Solves (H - wr * I) x = vr where all quantities are real.
 *
 * @param[in] H            Hessenberg matrix
 * @param[in] ldH          Leading dimension of H.
 * @param[in] Hnorms       Upper bounds of tiles Hij given by the partitioning p.
 * @param[in] p            Partitioning of vr.
 * @param[in] wr           Real eigenvalues.
 * @param[in,out] vr       Eigenvector matrix of size n-by-num_rhs.
 * @param[in] ldvr         Leading dimension of vr.
 * @param[in, out] vrnorms Matrix of size num_tile_rows x num_rhs. On entry,
 *                         vrnorms[k + num_rhs * tli] contains the upper bound
 *                         of eigenvector k for the tile row segment tli. On
 *                         exit, the upper bound of the solution.
 * @param[out] qwork       Workspace of size 2 * n * num_rhs. On exit, qwork holds
 *                         the Givens rotations reduce (H - wr * I) to triangular shape.
 * @param[out] R           Workspace of size 2 * n * num_tiles * nun_rhs. On exit,
 *                         R holds the crossover columns.
 *
 * See Algorithm 8 in https://arxiv.org/pdf/2101.05063.pdf for details.
 * */
void solve_Hessenberg_system_real_shift(
    const double *restrict H, const int ldH, double *restrict Hnorms,
    partitioning_t *p,
    const double *restrict wr, double *restrict vr, const int ldvr,
    double *restrict vrnorms, double *restrict qwork, double *restrict R);
#endif
