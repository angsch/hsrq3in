#include "solve-task.h"
#include "robust.h"
#include "reduce.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include <omp.h>


#define H(i,j) H[(i) + (j) * ldH]
#define vr(i,j) vr[(i) + (j) * ldvr]

int solve(
    const double *restrict cos, int ldcos, 
    const double *restrict sin, int ldsin,
    int n, const double *restrict H, int ldH, const double hleft,
    double *restrict Hnorm,
    const double *restrict R, int ldR,
    int nrhs, const double *restrict wr, const double smlnum,
    scaling_t *restrict scales,
    double *restrict vrnorms, double *restrict vr, int ldvr)
{
    // Compute the upper bound || H ||_oo, if required.
    if (*Hnorm < 0.0) {
        double Hnrm = dlange('I', n, n, H, ldH);
        #pragma omp atomic write
        *Hnorm = Hnrm;
    }

    int info = 0;

    double v[n];

    for (int k = 0; k < nrhs; k++) {
        // Extract the shift.
        double mu = wr[k];

        // Critical threshold to detect unsafe divisions.
        const double smin = fmax(DBL_EPSILON/2*fabs(mu), smlnum);

        // Locate the k-th eigenvector, corresponding scaling factor.
        double *y = vr + k * ldvr;
        scaling_t *beta = scales + k;

        // Locate Givens rotations.
        const double *c = cos + k * ldcos;
        const double *s = sin + k * ldsin;

        // Locate the right-most column.
        const double *r = R + k * ldR;

        // Prepare the right-most column.
        // Note that the shift mu has already been applied to r.
        memcpy(v, r, n * sizeof(double));

        // RQ-decomposition.
        for (int j = n - 1; j >= 1; j--) {
            ////////////////////////////////////////////////////////////////////
            // Solve R \ vr with backsubstitution, where R is implicitly formed.
            ////////////////////////////////////////////////////////////////////
            // y(j)   = y(j) / (c(j) * v(j) - s(j) * H(j,j-1));
            scaling_t phi;
            init_scaling_factor(1, &phi);
            info |= solve_1x1_real_system(
                smin, c[j] * v[j], s[j] * H(j,j-1), &y[j], &phi);
            update_global_scaling(beta, phi);

            // Scale remaining parts of the vector.
            scale_except(n, j, y, &phi);

            // y(j-1) = y(j-1) - y(j) * c(j) * v(j-1) + y(j) * s(j) * (H(j-1,j-1)-mu);
            phi = protect_double_update(fabs(v[j-1]), fabs(y[j]),
                fabs(H(j-1,j-1)-mu), fabs(y[j]), fabs(y[j-1]));
            scale(n, y, &phi);
            update_global_scaling(beta, phi);
            y[j-1] = y[j-1] - y[j] * c[j] * v[j-1] + y[j] * s[j] * (H(j-1,j-1)-mu);

            // y(1:i-2) = y(1:i-2) - y(i) * conj(c(i)) * v(1:i-2) + y(i) * conj(s(i)) * H(1:i-2,i-1);
            const double ynorm = vector_infnorm(j-1, y);
            const double vnorm = vector_infnorm(j-1, v);
            phi = protect_double_update(vnorm, fabs(y[j]), *Hnorm, fabs(y[j]), ynorm);
            scale(n, y, &phi);
            update_global_scaling(beta, phi);
            for (int i = 0; i < j - 1; i++) {
                y[i] = y[i] - y[j] * c[j] * v[i] + y[j] * s[j] * H(i,j-1);
            }

            ////////////////////////////////////////////////////////////////////
            // Prepare annihilation column for the next iteration.
            ////////////////////////////////////////////////////////////////////
            v[j-1] = c[j] * (H(j-1,j-1) - mu) + s[j] * v[j-1];
            for (int i = 0; i < j - 1; i++) {
                v[i] = c[j] * H(i,j-1) + s[j] * v[i];
            }
        }

        // Apply the final rotation to R(0,0).
        {
            v[0] = v[0] * c[0] - hleft * s[0];
            // y(0) = y(0) / v(0);
            scaling_t phi;
            init_scaling_factor(1, &phi);
            info |= solve_1x1_real_system(smin, v[0], 0.0, &y[0], &phi);
            update_global_scaling(beta, phi);

            // Scale remaining parts of the vector.
            scale(n-1, &y[1], &phi);
        }

    }

    return info;
}



int factor_and_solve_R11(
    double *restrict cos, int ldcos, 
    double *restrict sin, int ldsin,
    int n, const double *restrict H, int ldH, double *restrict Hnorm,
    const double *restrict R, int ldR,
    int nrhs, const double *restrict wr, const double smlnum,
    scaling_t *restrict scales,
    double *restrict vrnorms, double *restrict vr, int ldvr)
{
    // Compute the upper bound || H ||_oo, if required.
    if (*Hnorm < 0.0) {
        double Hnrm = dlange('I', n, n, H, ldH);
        #pragma omp atomic write
        *Hnorm = Hnrm;
    }


    int info = 0;

    double v[n];

    for (int k = 0; k < nrhs; k++) {
        // Extract the shift.
        double mu = wr[k];

        // Critical threshold to detect unsafe divisions.
        const double smin = fmax(DBL_EPSILON/2*fabs(mu), smlnum);

        // Locate the k-th eigenvector the its corresponding scaling factor.
        double *y = vr + k * ldvr;
        scaling_t *beta = scales + k;

        // Locate Givens rotations.
        double *c = cos + k * ldcos;
        double *s = sin + k * ldsin;

        // Locate the right-most column as the crossover column
        const double *r = R + k * ldR;

        // Prepare the right-most column.
        // Note that the shift mu has already been applied to r.
        memcpy(v, r, n * sizeof(double));

        // RQ-decomposition.
        for (int j = n - 1; j >= 1; j--) {
            givens(v[j], H(j,j-1), &c[j], &s[j]);
            // G = [  c  s ]
            //     [ -s  c ]

            ////////////////////////////////////////////////////////////////////
            // Solve R \ vr with backsubstitution, where R is implicitly formed.
            ////////////////////////////////////////////////////////////////////
            // y(j)   = y(j) / (c(j) * v(j) - s(j) * H(j,j-1));
            scaling_t phi;
            init_scaling_factor(1, &phi);
            info |= solve_1x1_real_system(
                smin, c[j] * v[j], s[j] * H(j,j-1), &y[j], &phi);
            update_global_scaling(beta, phi);

            // Scale remaining parts of the vector.
            scale_except(n, j, y, &phi);

            // y(j-1) = y(j-1) - y(j) * c(j) * v(j-1) + y(j) * s(j) * (H(j-1,j-1)-mu);
            phi = protect_double_update(fabs(v[j-1]), fabs(y[j]),
                fabs(H(j-1,j-1)-mu), fabs(y[j]), fabs(y[j-1]));
            scale(n, y, &phi);
            update_global_scaling(beta, phi);
            y[j-1] = y[j-1] - y[j] * c[j] * v[j-1] + y[j] * s[j] * (H(j-1,j-1)-mu);

            // y(1:i-2) = y(1:i-2) - y(i) * conj(c(i)) * v(1:i-2) + y(i) * conj(s(i)) * H(1:i-2,i-1);
            const double ynorm = vector_infnorm(j-1, y);
            const double vnorm = vector_infnorm(j-1, v);
            phi = protect_double_update(vnorm, fabs(y[j]), *Hnorm, fabs(y[j]), ynorm);
            scale(n, y, &phi);
            update_global_scaling(beta, phi);
            for (int i = 0; i < j - 1; i++) {
                y[i] = y[i] - y[j] * c[j] * v[i] + y[j] * s[j] * H(i,j-1);
            }

            ////////////////////////////////////////////////////////////////////
            // Prepare annihilation column for the next iteration.
            ////////////////////////////////////////////////////////////////////
            v[j-1] = c[j] * (H(j-1,j-1) - mu) + s[j] * v[j-1];
            for (int i = 0; i < j - 1; i++) {
                v[i] = c[j] * H(i,j-1) + s[j] * v[i];
            }
        }

        // y(0) = y(0) / v(0);
        scaling_t phi;
        init_scaling_factor(1, &phi);
        info |= solve_1x1_real_system(smin, v[0], 0.0, &y[0], &phi);
        update_global_scaling(beta, phi);

        // Scale remaining parts of the vector.
        scale(n-1, &y[1], &phi);
    }

#undef H
#undef vr

    return info;
}
