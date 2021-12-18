#include "inverse_iteration.h"
#include "partition.h"
#include "reduce.h"
#include "utils.h"
#include "shifted_Hessenberg_solve.h"
#include "robust.h"

#include <float.h>
#include <omp.h>
#include <assert.h>
#include <string.h>


#define Hnorms(i,j) Hnorms[(i) + (j) * num_tiles]
#define H(i,j) H[(i) + (j) * ldH]
#define vrnorms(rhs, tlrow) vrnorms[(rhs) + (tlrow) * nrhs]
#define vr(i,j) vr[(i) + (size_t)ldvr * (j)]

static void init_real_eigenvectors(
    int n, int nrhs, double *restrict vr, int ldvr,
    int num_tiles, double *restrict vrnorms, double eps3)
{
    assert(eps3 > 0);

    // Init eigenvectors.
    for (int rhs = 0; rhs < nrhs; rhs++)
        for (int i = 0; i < n; i++)
            vr(i,rhs) = eps3;

    // Record upper bounds.
    for (int i = 0; i < num_tiles; i++)
        for (int rhs = 0; rhs < nrhs; rhs++)
            vrnorms(rhs,i) = eps3;
}

static void new_real_starting_vector(
    int n, int nrhs, double *restrict vr, int ldvr,
    int num_tiles, double *restrict vrnorms, double eps3, int its)
{
    assert(eps3 > 0);
    assert(0 <= its && its < n);

    // See LAPACK 3.10 dlaein lines 344--353.
    const double tmp = eps3 / (sqrt(n) + 1.0);
    for (int rhs = 0; rhs < nrhs; rhs++) {
        vr(0,rhs) = eps3;
        for (int i = 1; i < n; i++)
            vr(i,rhs) = tmp;
        vr(n-1-its,rhs) -= eps3 * sqrt(n);
    }

    // Record upper bounds.
    const double ub = fmax(fmax(eps3, tmp), fabs(tmp - eps3 / sqrt(n)));
    for (int i = 0; i < num_tiles; i++)
        for (int rhs = 0; rhs < nrhs; rhs++)
            vrnorms(rhs,i) = ub;
}


static int check_convergence(int nrhs, int num_tiles, const double *restrict vrnorms)
{
    int converged = 1;

    for (int rhs = 0; rhs < nrhs; rhs++) {
        if (vrnorms(rhs,0) == 0.0) {
            converged = 0;
            break;
        }
    }

    return converged;
}


static int separate_nonconverged(
    int nrhs, int num_tiles, const double *restrict vrnorms,
    double *restrict wr, int n, double *restrict vr, int ldvr)
{
    // Count the converged eigenvectors.
    int num_converged = 0;
    for (int rhs = 0; rhs < nrhs; rhs++)
        if (vrnorms(rhs,0) == 0.0)
            num_converged++;

    // This routine should only be reached if a new starting vector is needed.
    assert(num_converged < nrhs);

    // Sort wr, vr into converged and non-converged eigenvalue-eigenvector pairs.
    double *restrict sorted_wr = (double *) malloc(nrhs * sizeof(double));
    double *restrict sorted_vr = (double *) malloc(ldvr * nrhs * sizeof(double));
#define sorted_vr(i,j) sorted_vr[(i) + (j) * (size_t)ldvr]
    int pos_converged = 0;
    int pos_non_converged = num_converged;

    int num_nonconverged = nrhs - num_converged;

    for (int rhs = 0; rhs < nrhs; rhs++) {
        if (vrnorms(rhs,0) == 0.0) {
            sorted_wr[pos_non_converged] = wr[rhs];
            memcpy(&sorted_vr(0,pos_non_converged), &vr(0,rhs), n * sizeof(double));
            pos_non_converged++;
        }
        else {
            sorted_wr[pos_converged] = wr[rhs];
            memcpy(&sorted_vr(0,pos_converged), &vr(0,rhs), n * sizeof(double));
            pos_converged++;
        }
    }
    assert(pos_converged == num_converged);
    assert(pos_non_converged == n);

    // Copy back wr := sorted_wr.
    memcpy(wr, sorted_wr, nrhs * sizeof(double));

    // Copy back vr := vr_sorted.
    for (int rhs = 0; rhs < nrhs; rhs++)
        memcpy(&vr(0,rhs), &sorted_vr(0,rhs), n * sizeof(double));

    // Clean up.
    free(sorted_wr);
    free(sorted_vr);
#undef sorted_vr

    return num_converged;
}


void tiled_inverse_iteration(
    int n, const double *restrict H, int ldH,
    int nrhs, double *restrict wr,
    int tlsz, int rhs_tlsz, double *restrict vr, int ldvr)
{
    // Partition H.
    int num_tiles = (n + tlsz - 1) / tlsz;
    int *first_row = (int *) malloc((num_tiles + 1) * sizeof(int));
    partition(n, num_tiles, tlsz, first_row);

    // Allocate workspace.
    double *Hnorms = (double *) malloc(num_tiles * num_tiles * sizeof(double));
    const int max_num_elems = 44000*44000;
    int batch_size = max(rhs_tlsz, max_num_elems / ( n * 2 * num_tiles));
    if (batch_size % 2 == 1) {
        batch_size--;
    }
    double *R = (double *) malloc((size_t)n * 2 * num_tiles * batch_size * sizeof(double));
    double *qwork = (double *) malloc((size_t)n * batch_size * 3 * sizeof(double));

    // Init upper bounds || Hij ||_oo.
    for (int tli = 0; tli < num_tiles; tli++)
        for (int tlj = 0; tlj < num_tiles; tlj++)
            Hnorms(tli, tlj) = -1.0;

    // The starting vectors require || H ||_oo.
    double Hnrm = dlange('I', n, n, H, ldH);

    // Prepare initial value of starting vector.
    const double eps = DBL_EPSILON / 2;
    const double eps3 = Hnrm * eps;

    if (nrhs <= batch_size) { // Non-batched mode.
        int num_rhs_tiles = (nrhs + rhs_tlsz - 1) / rhs_tlsz;
        int *first_col = (int *) malloc((num_rhs_tiles + 1) * sizeof(int));
        partition(nrhs, num_rhs_tiles, rhs_tlsz, first_col);
        partitioning_t p = {.num_tile_rows = num_tiles,
                            .num_tile_cols = num_rhs_tiles,
                            .first_row = first_row,
                            .first_col = first_col};

        double *vrnorms = (double *) malloc(num_tiles * nrhs * sizeof(double));

        init_real_eigenvectors(n, nrhs, vr, ldvr, num_tiles, vrnorms, eps3);

        int converged = 0;
        int its = 0;

        do {
            int num_converged = 0;

            solve_Hessenberg_system_real_shift(
                H, ldH, Hnorms, &p, &wr[num_converged],
                &vr(0,num_converged), ldvr, vrnorms,
                qwork, R);

            // The convergence status is propagated in vrnorms(:,0).
            converged = check_convergence(nrhs, num_tiles, vrnorms);

            if (!converged) { // Rare event.
                // Sort eigenvalues that have not converged.
                num_converged = separate_nonconverged(nrhs, num_tiles,
                                                      vrnorms, wr, n, vr, ldvr);

                // Prepare a new starting vector.
                nrhs = nrhs - num_converged;
                assert(nrhs > 0);
                new_real_starting_vector(n, nrhs, &vr(0,num_converged), ldvr,
                                         num_tiles, vrnorms, eps3, its);

                // Update partitioning.
                num_rhs_tiles = (nrhs + rhs_tlsz - 1) / rhs_tlsz;
                partition(nrhs, num_rhs_tiles, rhs_tlsz, first_col);
                p.num_tile_cols = num_rhs_tiles;

                its++;
            }
        } while(!converged && its < n);

        free(first_col);
        free(vrnorms);
    } else { // Batched mode.
        int all_converged = 0;
        double *converged = (double *) malloc(nrhs * sizeof(double));

        for (int rhs = 0; rhs < nrhs; rhs += batch_size) {
            int b = min(batch_size, nrhs - rhs);
            int num_rhs_tiles = (b + rhs_tlsz - 1) / rhs_tlsz;
            int *first_col = (int *) malloc((num_rhs_tiles + 1) * sizeof(int));
            partition(b, num_rhs_tiles, rhs_tlsz, first_col);
            partitioning_t p = {.num_tile_rows = num_tiles,
                                .num_tile_cols = num_rhs_tiles,
                                .first_row = first_row,
                                .first_col = first_col};

            double *vrnorms = (double *) malloc(num_tiles * nrhs * sizeof(double));

            // Init the current batch of eigenvectors.
            init_real_eigenvectors(
                n, b, &vr(0,rhs), ldvr, num_tiles, &vrnorms(rhs,0), eps3);

            solve_Hessenberg_system_real_shift(
                H, ldH, Hnorms, &p, &wr[rhs],
                &vr(0,rhs), ldvr, &vrnorms(rhs,0), qwork, R);

            // The convergence status is propagated in vrnorms(:,0).
            all_converged = check_convergence(b, num_tiles, &vrnorms(rhs,0));
            if(all_converged) {
                for (int j = 0; j < b; j++)
                    converged[rhs + j] = 1;
            }
            else { // !all_converged
                // Record status, non-converged eigenvectors will be treated
                // together later.
                for (int j = 0; j < b; j++) {
                    if (vrnorms(rhs + j,0) == 0.0)
                        converged[rhs + j] = 0.0;
                    else
                        converged[rhs + j] = 1.0;
                }
            }

            free(first_col);
            free(vrnorms);
        } // for rhs

        // Check if all eigenvectors have converged.
        int num_converged = 0;
        for (int j = 0; j < nrhs; j++)
            if (converged[j])
                num_converged++;

        // Treat all non-converged eigenvectors.
        if (num_converged < nrhs) {
            // Sort eigenvalues that have not converged. Make sure that there
            // is enough workspace available to execute the sorting.
            free(R);
            separate_nonconverged(nrhs, num_tiles, converged, wr, n, vr, ldvr);

            // Reallocate the workspace.
            R = (double *) malloc((size_t)n * 2 * num_tiles * batch_size * sizeof(double));

            all_converged = 0;
            int its = 0;

            do {
                // Prepare a new starting vector.
                nrhs = nrhs - num_converged;
                assert(nrhs > 0);
                double *vrnorms = (double *) malloc(num_tiles * nrhs * sizeof(double));


                new_real_starting_vector(n, nrhs, &vr(0,num_converged), ldvr,
                                         num_tiles, vrnorms, eps3, its);

                // Repartition.
                int num_rhs_tiles = (nrhs + rhs_tlsz - 1) / rhs_tlsz;
                int *first_col = (int *) malloc((num_rhs_tiles + 1) * sizeof(int));
                partition(nrhs, num_rhs_tiles, rhs_tlsz, first_col);
                partitioning_t p = {.num_tile_rows = num_tiles,
                                    .num_tile_cols = num_rhs_tiles,
                                    .first_row = first_row,
                                    .first_col = first_col};

                // Solve.
                solve_Hessenberg_system_real_shift(
                    H, ldH, Hnorms, &p, &wr[num_converged],
                    &vr(0,num_converged), ldvr, vrnorms,
                    qwork, R);

                its++;

                // The convergence status is propagated in vrnorms(:,0).
                all_converged = check_convergence(nrhs, num_tiles, vrnorms);

                if (!all_converged) {
                    // Sort eigenvalues that have not converged.
                    num_converged += separate_nonconverged(
                        nrhs, num_tiles, vrnorms, &wr[num_converged], n,
                        &vr(0,num_converged), ldvr);

                    // Prepare a new starting vector.
                    nrhs = nrhs - num_converged;
                    assert(nrhs > 0);
                    new_real_starting_vector(n, nrhs, &vr(0,num_converged), ldvr,
                                             num_tiles, vrnorms, eps3, its);

                    // Update partitioning.
                    num_rhs_tiles = (nrhs + rhs_tlsz - 1) / rhs_tlsz;
                    partition(nrhs, num_rhs_tiles, rhs_tlsz, first_col);
                    p.num_tile_cols = num_rhs_tiles;
                }

                // Clean up.
                free(vrnorms);
                free(first_col);
            } while(!all_converged && its < n);

        }
        free(converged);
    }

    // Clean up.
    free(first_row);
    free(Hnorms);
    free(R);
    free(qwork);
}
