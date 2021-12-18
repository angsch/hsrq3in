#include "shifted_Hessenberg_solve.h"
#include "partition.h"
#include "reduce.h"
#include "robust.h"
#include "update-task.h"
#include "solve-task.h"

#include <float.h>
#include <string.h>
#include <omp.h>


// Scaling constants used in LAPACK 3.10 in dnrm2. See also the reference
// implementation attached to https://dl.acm.org/doi/10.1145/3061665.
const double tsml = 0.14916681462400413487e-153;
const double tbig = 0.19979190722022350281E+147;
const double ssml = 0.44989137945431963828E+162;
const double sbig = 0.11113793747425387417E-161;


static void init_2D_lock_grid(int m, int n, omp_lock_t *lock)
{
#define lock(i,j) lock[(i) + m * (j)]

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            omp_init_lock(&lock(i,j));

#undef lock
}


static int converged(int n, double ynrm)
{
    // Converged if || x ||_2 > 0.1 / sqrt(n).
    //
    // This is in line with the convergence criterion proposed by Varah
    // || y^(1) / y^(0) ||_2 > 1 / (c * eps).
    return ynrm > 0.1 / sqrt(n);
}


static void robustTiledSolve(
    const double *restrict H, int ldH, double *restrict Hnorms,
    partitioning_t *p, const double *restrict const wr,
    double *restrict *restrict Rtildes,
    double *restrict *restrict c, double *restrict *restrict s,
    double *restrict vr, int ldvr, double *restrict vrnorms)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int num_rhs_tiles = p->num_tile_cols;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    int n = first_row[num_tiles];
    int nrhs = first_col[num_rhs_tiles];

    // Partition vr.
    double ***vr_tiles = malloc(num_tiles * sizeof(double **));
    for (int i = 0; i < num_tiles; i++) {
        vr_tiles[i] = malloc(num_rhs_tiles * sizeof(double *));
    }
    partition_matrix(vr, ldvr, p, vr_tiles);

    // Allocate locks for tiled vr to synchronize the updates.
    omp_lock_t lock[num_tiles][num_rhs_tiles];
    init_2D_lock_grid(num_tiles, num_rhs_tiles, &lock[0][0]);

    // Define threshold when close eigenvalues shall be perturbed.
    const double eps = DBL_EPSILON/2;
    const double smlnum = fmax(2*DBL_MIN, DBL_MIN*((double)n/eps));

    // Workspace for overflow protection.
    scaling_t *scales, *smin;
    scales = (scaling_t *) malloc(num_tiles * nrhs * sizeof(scaling_t));
    smin = (scaling_t *) malloc(nrhs * sizeof(scaling_t));

#define H(i,j) H[(i) + (j) * ldH]
#define vr(i,j) vr[(i) + (j) * (size_t)ldvr]
#define scales(col, blkrow) scales[(col) + (blkrow) * nrhs]
#define vrnorms(col, blkrow) vrnorms[(col) + (blkrow) * nrhs]
#define Hnorms(i,j) Hnorms[(i) + (j) * num_tiles]
#define c(tli,tlk) c[(tli) + num_tiles * (tlk)]
#define s(tli,tlk) s[(tli) + num_tiles * (tlk)]
#define Rtildes(tli,tlk,tlj) Rtildes[(tli) + num_tiles * (tlk) + (size_t)num_tiles * num_rhs_tiles * (tlj)]

    // Initialize all tile-local scaling factor with 1 (or 0 if integer scaling
    // factors are used).
    {
        int nrhs = first_col[num_rhs_tiles];
        init_scaling_factor(num_tiles * nrhs, scales);
    }


    #pragma omp parallel shared(vr_tiles) shared(vrnorms) shared(c) shared(s) shared(Rtildes) shared(first_row) shared(first_col) shared(scales) shared(smin) shared(lock)
    #pragma omp single nowait
    {
        // Loop over tile columns.
        for (int tlj = num_tiles - 1; tlj >= 0; tlj--) {
            //////////////////////////////
            // Solve
            //////////////////////////////
            for (int tlk = 0; tlk < num_rhs_tiles; tlk++) {
                if (tlj == 0) {
                    // Process top-left corner of the Hessenberg matrix: Fuse
                    // reduction, triangular solve and backtransform.
                    #pragma omp task \
                      depend(in: vr_tiles[1:num_tiles][tlk]) \
                      depend(inout: vr_tiles[0][tlk])
                    {
                        int num_rows = first_row[tlj+1] - first_row[tlj];
                        int num_cols = first_col[tlk+1] - first_col[tlk];
                        int k = first_col[tlk];

                        factor_and_solve_R11(
                            c(tlj,tlk), num_rows, s(tlj,tlk), num_rows,
                            num_rows, &H(0,0), ldH, &Hnorms(0,0),
                            Rtildes(tlj,tlk,tlj), num_rows,
                            num_cols, wr + k, smlnum,
                            &scales(k,0), &vrnorms(k,0), &vr(0,k), ldvr);

                        // Compute the most constraining scaling factor.
                        reduce_scaling_factors(num_cols, num_tiles,
                            &scales(k,0), first_col[num_rhs_tiles], smin + k);

                        for (int kk = 0; kk < num_cols; kk++) {
                            // Absolute column index.
                            const int rhs = first_col[tlk] + kk;

                            double tau1, tau2;

                            // As of LAPACK 3.10, the computation of the 2-norm
                            // has changed and sorts vector entries into 3
                            // accumulators.
                            double asml = 0.0;
                            double amed = 0.0;
                            double abig = 0.0;

                            double alpha = compute_upscaling(smin[rhs], scales(rhs,0));

                            for (int tli = 0; tli < num_tiles; tli++) {
                                // Locate vector.
                                double *y = &vr(first_row[tli],rhs);

                                const int ldc = first_row[tli + 1] - first_row[tli];
                                const int lds = ldc;
                                const double *sk = s(tli,tlk) + kk * lds;
                                const double *ck = c(tli,tlk) + kk * ldc;
                                const int nrows = first_row[tli+1] - first_row[tli];

                                alpha = compute_upscaling(smin[rhs], scales(rhs, tli));

                                int start;
                                if (tli == 0) {
                                    start = 1;
                                    tau1 = alpha * y[0];
                                }
                                else {
                                    start = 0;
                                }

                                //////////////////////////////
                                // Backtransform.
                                //////////////////////////////
                                for (int i = start; i < nrows; i++){
                                    tau2 = alpha * y[i];
                                    y[i-1] = ck[i] * tau1 - sk[i] * tau2;
                                    tau1 = ck[i] * tau2 + sk[i] * tau1;

                                    // Compute 2-norm of y alongisde with the
                                    // backtransform using a simplified version
                                    // of LAPACK 3.10's dnrm2 routine.
                                    const double abs = fabs(y[i-1]);
                                    if (abs > tbig)
                                        abig += (abs*sbig) * (abs*sbig);
                                    else if (abs < tsml)
                                        asml += (abs*ssml) * (abs*ssml);
                                    else
                                        amed += abs*abs;

                                }

                            }
                            vr(n-1,rhs) = tau1;

                            // The final entry is not covered in the norm computation
                            // in the loop.
                            {
                                const double abs = fabs(vr(n-1,rhs));
                                if (abs > tbig)
                                    abig += (abs*sbig) * (abs*sbig);
                                else if (abs < tsml)
                                    asml += (abs*ssml) * (abs*ssml);
                                else
                                    amed += abs*abs;
                            }

                            // Compute the 2-norm by aggregating the accumulators
                            // as done in LAPACK 3.10.
                            double ynrm = 0.0;
                            double scale = 1.0;
                            if (abig > 0.0) {
                                ynrm = abig + (amed * sbig) * sbig;
                                scale = 1.0 / sbig;
                            }
                            else if (asml > 0.0) {
                                if (amed > 0.0) {
                                    amed = sqrt(amed);
                                    asml = sqrt(asml) / ssml;
                                    double ymin = 0.0, ymax = 0.0;
                                    if (asml > amed) {
                                        ymin = amed;
                                        ymax = asml;
                                    }
                                    else {
                                        ymin = asml;
                                        ymax = amed;
                                    }
                                    ynrm = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
                                }
                                else {
                                    scale = 1.0 / ssml;
                                    ynrm = asml;
                                }
                            }
                            else {
                                ynrm = amed;
                            }
                            ynrm = scale * sqrt(ynrm);

                            // Convergence check.
                            if (!converged(n, ynrm)) {
                                printf("not converged\n");
                                // Propagate non-convergence.
                                vrnorms(rhs, 0) = 0.0;
                            }
                            else {
                                // Converged! Normalize eigenvector.
                                for (int i = 0; i < n; i++)
                                    vr(i,rhs) /= ynrm;
                                vrnorms(rhs, 0) = 1.0;
                            }
                        }
                    }
                }
                else {
                    // Standard case: triangular solve on a diagonal tile
                    // using previously computed Givens rotations.
                    #pragma omp task \
                      depend(in: vr_tiles[tlj+1:num_tiles][tlk]) \
                      depend(inout: vr_tiles[tlj][tlk])
                    {
                        int num_rows = first_row[tlj+1] - first_row[tlj];
                        int k = first_col[tlk];
                        int num_cols = first_col[tlk+1] - first_col[tlk];

                        // Find start index (l = left) of the diagonal block H(tlj,tlj).
                        const int l = first_row[tlj];

                        // Generate standard diagonal block (tlj = 2, ..., num_tiles-1)
                        // and solve the triangular system Rii \ vr.
                        solve(c(tlj,tlk), num_rows, s(tlj,tlk), num_rows,
                              num_rows, &H(l,l), ldH, H(l,l-1), &Hnorms(tlj,tlj),
                              Rtildes(tlj,tlk,tlj), num_rows,
                              num_cols, wr + k, smlnum,
                              &scales(k,tlj), &vrnorms(k,tlj), &vr(l,k), ldvr);
                    }
                }
            } // for tlk

            //////////////////////////////
            // Update without using R.
            //////////////////////////////
            if (tlj > 0) {
                for (int tlk = 0; tlk < num_rhs_tiles; tlk++) {
                    for (int tli = tlj - 1; tli >= 0; tli--) {
                        #pragma omp task \
                          depend(in: vr_tiles[tlj][tlk]) \
                          depend(inout: vr_tiles[tli][tlk])
                        {
                            int mm = first_row[tli+1] - first_row[tli];
                            int nn = first_col[tlk+1] - first_col[tlk];
                            int kk = first_row[tlj+1] - first_row[tlj];

                            const int ldRtildes = mm;

                            // Mark if the update processes the tile above the
                            // subdiagonal.
                            int subdiag = 0;
                            if (tli == tlj - 1)
                                subdiag = 1;

                            tile_update(mm, nn, kk,
                                   &vr(first_row[tli],first_col[tlk]), ldvr,
                                   &vrnorms(first_col[tlk],tli),
                                   &scales(first_col[tlk],tli),
                                   &H(first_row[tli], first_row[tlj] - 1),
                                   &H(first_row[tli],first_row[tlj]), ldH,
                                   &Hnorms(tli,tlj),
                                   wr + first_col[tlk],
                                   &lock[tli][tlk],
                                   &vr(first_row[tlj],first_col[tlk]), ldvr,
                                   &scales(first_col[tlk],tlj),
                                   s(tlj,tlk), kk,
                                   c(tlj,tlk), kk,
                                   Rtildes(tli,tlk,tlj), ldRtildes,
                                   subdiag);
                        }
                    } // for tli
                } // for tlk
            } // if
        } // for tlj
    } // parallel

#undef H
#undef vr
#undef scales
#undef vrnorms
#undef Hnorms
#undef c
#undef s
#undef Rtildes

    // Clean up.
    for (int i = 0; i < num_tiles; i++) {
        free(vr_tiles[i]);
    }
    free(vr_tiles);
    free(scales);
    free(smin);
}


void solve_Hessenberg_system_real_shift(
    const double *restrict H, const int ldH, double *restrict Hnorms,
    partitioning_t *p,
    const double *restrict wr, double *restrict vr, const int ldvr,
    double *restrict vrnorms,
    double *restrict qwork,   // c = (n * num_rhs); s = (n * num_rhs);
    double *restrict R)
{
    // Extract the partitioning.
    const int num_tiles = p->num_tile_rows;
    const int num_rhs_tiles = p->num_tile_cols;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    int n = first_row[num_tiles];
    int num_rhs = first_col[num_rhs_tiles];

    // Partition Rtildes in tile layout.
    double *Rtildes[num_tiles * num_tiles * num_rhs_tiles];
#define Rtildes(tli,tlk,tlj) Rtildes[(tli) + num_tiles * (tlk) + (size_t)num_tiles * num_rhs_tiles * (tlj)]
    double *tile_ptr = R;
    for (int tlj = 0; tlj < num_tiles; tlj++) {
        for (int tlk = 0; tlk < num_rhs_tiles; tlk++) {
            for (int tli = 0; tli < num_tiles; tli++) {
                // Compute tile dimensions.
                int num_rows = first_row[tli + 1] - first_row[tli];
                int num_cols = first_col[tlk + 1] - first_col[tlk];
                int tile_size = num_rows * num_cols;

                // Record start address of tile.
                Rtildes(tli, tlk, tlj) = tile_ptr;
                tile_ptr += tile_size;
            }
        }
    }

    // Partition sin, cos in tile layout.
    double *c[num_tiles * num_rhs_tiles];
    double *s[num_tiles * num_rhs_tiles];
#define c(tli,tlk) c[(tli) + num_tiles * (tlk)]
#define s(tli,tlk) s[(tli) + num_tiles * (tlk)]
    tile_ptr = qwork;
    for (int tli = 0; tli < num_tiles; tli++) {
        for (int tlk = 0; tlk < num_rhs_tiles; tlk++) {
            // Compute tile dimensions.
            int num_rows = first_row[tli + 1] - first_row[tli];
            int num_cols = first_col[tlk + 1] - first_col[tlk];
            int tile_size = num_rows * num_cols;

            // Record start address of tiles.
            c(tli,tlk) = tile_ptr;
            s(tli,tlk) = tile_ptr + n * num_rhs;
            tile_ptr += tile_size;
        }
    }

    memset(qwork, 0.0, 3 * n * num_rhs * sizeof(double));

    // Solve shifted Hessenberg system.
    tiledReduce(H, ldH, p, wr, Rtildes, c, s);
    robustTiledSolve(H, ldH, Hnorms, p, wr, Rtildes, c, s, vr, ldvr, vrnorms);
}
