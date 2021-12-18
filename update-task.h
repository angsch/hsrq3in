#ifndef UPDATE_TASK_H
#define UPDATE_TASK_H

#include "utils.h"
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
    const int applyShift);


#endif
