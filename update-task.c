#include "update-task.h"
#include "utils.h"
#include "robust.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include <omp.h>


void tile_update(
    int m, int n, int k,
    double *restrict Y, int ldY,
    double *restrict Ynorms, scaling_t *restrict Yscales,
    const double *restrict h,
    const double *restrict H, int ldH, double *restrict Hnorm,
    const double *restrict wr,
    omp_lock_t *lock,
    const double *restrict X, int ldX,
    const scaling_t *restrict Xscales,
    const double *restrict s, int lds, const double *restrict c, int ldc,
    const double *restrict Rtildes, const int ldRtildes,
    const int applyShift)
{
    //    Y := Y -  H   *   X
    // (m x n)   (m x k) (k x n)

    while (!omp_test_lock(lock)) {
        #pragma omp taskyield
        ;
    }

    // Aliases.
    int num_rhs = n;

    // Status flag if X has to be rescaled.
    int rescale_X = 0;

    // Workspace to store locally computed scaling factors.
    scaling_t tmp_scales[n];

#define H(i,j) H[(i) + (j) * (size_t)ldH]
#define X(i,k) X[(i) + (k) * (size_t)ldX]
#define Y(i,k) Y[(i) + (k) * (size_t)ldY]
#define s(k,rhs) s[(k) + (rhs) * (size_t)lds]
#define c(k,rhs) c[(k) + (rhs) * (size_t) ldc]
#define Rtildes(i,rhs) Rtildes[(i) + (size_t)ldRtildes * (rhs)]

    ////////////////////////////////////////////////////////////////////////////
    // Compute the upper bound || H ||_oo, if required.
    ////////////////////////////////////////////////////////////////////////////

    if (*Hnorm < 0.0) {
        double Hnrm = dlange('I', m, k, H, ldH);
        #pragma omp atomic write
        *Hnorm = Hnrm;
    }


    ////////////////////////////////////////////////////////////////////////////
    // Compute a consistent scaling.
    ////////////////////////////////////////////////////////////////////////////
    for (int rhs = 0; rhs < num_rhs; rhs++) {
        if (Yscales[rhs] < Xscales[rhs]) {
            // Mark X for scaling. Physical rescaling is deferred.
            rescale_X = 1;
        }
        else if (Xscales[rhs] < Yscales[rhs]) {
            // The common scaling factor is Xscales[rhs].
            //const double a = Xscales[rhs] / Yscales[rhs];
            const double a = compute_upscaling(Xscales[rhs], Yscales[rhs]);

            // Rescale Y.
            for (int i = 0; i < m; i++)
                Y(i,rhs) = a * Y(i,rhs);

            // Update norm using norm(a * Y) = a * norm(Y).
            Ynorms[rhs] = a * Ynorms[rhs];

            // Update global scaling of Y.
            Yscales[rhs] = Xscales[rhs];
        }
    }


    ////////////////////////////////////////////////////////////////////////////
    // Part 1.
    // y(1:l-1) = y(1:l-1) - (H11-mu*I) * (Q21^T*y(l:r))
    //          = y(1:l-1) - (H11-mu*I) *    ytilde
    // where ytilde is given as the top entry of
    //     G_l^T * [0;    =  [c(l) -s(l);   * [0   ;    = [ ytilde;
    //              y(l)]     s(l)  c(l) ]     y(l) ]         *    ]
    ////////////////////////////////////////////////////////////////////////////

    // Norm of column h.
    const double hnrm = vector_infnorm(m, h);

    for (int rhs = 0; rhs < num_rhs; rhs++) {
        // Extract the shift.
        const double mu = wr[rhs];
        double ytilde;

        // Apply Givens rotations to potentially downscaled rhs.
        if (Yscales[rhs] < Xscales[rhs]) {
            // The common scaling factor is Yscales[rhs].
            const double a = Yscales[rhs] / Xscales[rhs];
            ytilde = -s(0,rhs) * (a * X(0,rhs));
            rescale_X = 1;
        }
        else {
            ytilde = -s(0,rhs) * X(0,rhs);
        }

        // Compute a scaling factor so that the linear updates
        // cannot overflow.
        const scaling_t alpha = protect_update(
            fmax(hnrm, fabs(mu)), fabs(ytilde), Ynorms[rhs]);
        const double a = convert_scaling(alpha);

        if (a != 1.0) {
            // Execute the linear update safely.
            for (int i = 0; i < m; i++)
                Y(i,rhs) = (a * Y(i,rhs)) - h[i] * (a * ytilde);
            if (applyShift) {
                Y(m-1,rhs) = (a * Y(m-1,rhs)) + mu * (a * ytilde);
            }
        }
        else {
            // Execute the linear update safely.
            for (int i = 0; i < m; i++)
                Y(i,rhs) = Y(i,rhs) - h[i] * ytilde;
            if (applyShift) {
                Y(m-1,rhs) = Y(m-1,rhs) + mu * ytilde;
            }
        }

        // Update the global scaling of Y.
#ifdef INTSCALING
        Yscales[rhs] = Yscales[rhs] + alpha;
#else
        Yscales[rhs] = Yscales[rhs] * alpha;
#endif

        // Recompute Ynorms.
        Ynorms[rhs] = vector_infnorm(m, Y + ldY * rhs);
    }



    ////////////////////////////////////////////////////////////////////////////
    // Part 2.
    // y(1:l-1) = y(1:l-1) - H12 *             Q22^T           * y(l:r)
    //          = y(1:l-1) - H12 * G_n^T*...*G_{r+1}^T*G_{r}^T*...*G_{l}^T * y(l:r)
    //          = y(1:l-1) - (H12*G_n^T*...*G_{r+1}^T) * (G_{r}^T*...*G_{l}^T*y(l:r))
    //          = y(1:l-1) -        H12tilde           *    ytilde2
    ////////////////////////////////////////////////////////////////////////////
    double *Ytilde2 = 
        (double *) _mm_malloc((size_t)k * num_rhs * sizeof(double), ALIGNMENT);
#define Ytilde2(j,rhs) Ytilde2[(j) + (rhs) * (size_t)k]

    ////////////////////////////////////////////////////////////////////////////
    // Apply a subset of Givens rotations to the right-hand side.
    ////////////////////////////////////////////////////////////////////////////
    // Note that this computation is redundantly executed across update tasks.
    for (int rhs = 0; rhs < num_rhs; rhs++) {

        // Copy rhs to scratch buffer.
        if (rescale_X) {
            if (Yscales[rhs] < Xscales[rhs]) {
                // The common scaling factor is Yscales[k]. Note that Yscales[k]
                // already includes potential scalings from Part 1.
                const double a = compute_upscaling(Yscales[rhs], Xscales[rhs]);
                for (int i = 0; i < k; i++)
                    Ytilde2(i,rhs) = a * X(i,rhs);

            }
        }
        else {
            // No scaling is required.
            memcpy(&Ytilde2(0,rhs), &X(0,rhs), k * sizeof(double));
        }

        // Apply G_l.
        Ytilde2(0,rhs) = c(0,rhs) * Ytilde2(0,rhs);

        // Apply Givens rotations G_{l+1}, ..., G_r.
        for (int j = 1; j < k; j++) {
            const double tau1 = Ytilde2(j-1,rhs);
            const double tau2 = Ytilde2(j  ,rhs);
            Ytilde2(j-1,rhs) = c(j,rhs) * tau1 - s(j,rhs) * tau2;
            Ytilde2(j  ,rhs) = s(j,rhs) * tau1 + c(j,rhs) * tau2;
        }
    } // end for


    ////////////////////////////////////////////////////////////////////////////
    // Compute scalings.
    ////////////////////////////////////////////////////////////////////////////

    // Compute the norm of Ytilde2 (or: a * Xnorms * 4 -- a growth by a factor of 4 is the maximum possible growth)
    double ytilde2nrm[num_rhs];
    for (int rhs = 0; rhs < num_rhs; rhs++) {
        ytilde2nrm[rhs] = vector_infnorm(k, &Ytilde2(0,rhs));
    }

    // Compute a scaling factor such that the linear update is safe.
    int status = protect_multi_rhs_update_real(
        ytilde2nrm, num_rhs, *Hnorm, Ynorms, tmp_scales);

    // Apply the scaling factor, if necessary.
    if (status == RESCALE) {
        for (int rhs = 0; rhs < num_rhs; rhs++) {
            scale(k, &Ytilde2(0,rhs), tmp_scales + k);
            scale(m, Y + ldY * rhs, tmp_scales + rhs);
#ifdef INTSCALING
            Yscales[rhs] = Yscales[rhs] + tmp_scales[rhs];
#else
            Yscales[rhs] = Yscales[rhs] * tmp_scales[k];
#endif
        }
    }


    ////////////////////////////////////////////////////////////////////////////
    // Compute robustly
    //     Y(1:l-1,:) = Y(1:l-1,:) - H(1:l-1,l:r-1) * Ytilde2(l:r-1,:)
    ////////////////////////////////////////////////////////////////////////////
    dgemm('N', 'N', m, n, k-1, -1.0, H, ldH, Ytilde2, k, 1.0, Y, ldY);


    ////////////////////////////////////////////////////////////////////////////
    // Rank-1 perturbation from shift.
    ////////////////////////////////////////////////////////////////////////////
    for (int rhs = 0; rhs < num_rhs; rhs++) {
        for (int i = 0; i < m; i++) {
            Y(i,rhs) = Y(i,rhs) - Rtildes(i,rhs) * Ytilde2(k-1,rhs);
        }
    }

    // Recompute and record Ynorms.
    for (int rhs = 0; rhs < num_rhs; rhs++) {
        Ynorms[rhs] = vector_infnorm(m, Y + ldY * rhs);
    }

    omp_unset_lock(lock);

#undef Ytilde2
    _mm_free(Ytilde2);


#undef H
#undef X
#undef Y
#undef s
#undef c
#undef Rtildes
}
