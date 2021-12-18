#ifndef ROBUST_H
#define ROBUST_H

#include "utils.h"


///
/// @brief Initializes all scaling factors with the neutral element.
///
/// @param[in] n
///         Number of entries in alpha. n >= 0.
///
/// @param[out] alpha
///         Vector of length n. On exit, all entries equal the neutral element.
///
void init_scaling_factor(
    int n, scaling_t *restrict const alpha);


///
/// @brief Scales a vector with a scalar.
///
/// @param[in] n
///         Number of entries in x. n >= 0.
///
/// @param[in,out] x
///         Vector of length n. On exit, x := beta * x.
///
/// @param[in] beta
///         Pointer to a scalar.
///
void scale(
    int n, double *restrict const x, const scaling_t *beta);


///
/// @brief Computes the column-wise minimum for several right-hand sides.
///
/// Given a matrix of scaling factors, sized num_rhs x num_tile_rows, this
/// routine computes the column-wise minimum:
///
///    smin[k] = min_{0 <= i <= num_tile_rows-1} ( scales(k, i) )
///
///
/// @param[in] num_rhs
///         Number of rows of scales, length of smin. num_rhs >= 0.
///
/// @param[in] num_tile_rows
///         Number of columns of scales. num_tile_rows >= 0.
///
/// @param[in] scales
///         A matrix of scaling factors with dimensions num_rhs x num_tile_rows.
///
/// @param[in] ldscales
///         Leading dimension of scales. ldscales >= num_rhs.
///
/// @param[out] smin
///         A vector of length num_rhs containing the most constraining scaling.

void reduce_scaling_factors(
    int num_rhs, int num_tile_rows,
    const scaling_t *restrict const scales, const int ldscales,
    scaling_t *restrict const smin);



///
/// @brief Combines two scalars.
///
/// @param[in,out] global
///         Pointer to a scalar. On exit the combined scalar of global and phi.
///
/// @param[in] phi
///         A scalar scaling factor.
///
void update_global_scaling(
    scaling_t *global, scaling_t phi);


///
/// @brief Compute ratio between alpha_min and alpha for upscaling.
///
/// @param[in] alpha_min
///         The smallest scalar.
///
/// @param[in] alpha
///         A scalar.
///
/// @return alpha_min / alpha
///
double compute_upscaling(
    scaling_t alpha_min, scaling_t alpha);


///
/// @brief Converts a scaling to a double-precision scaling factor.
///
/// @param[in] alpha
///         A scalar.
///
/// @return The scalar alpha converted to double-precision.
///
double convert_scaling(scaling_t alpha);



///
/// @brief Computes scaling such that the update y := y + t x cannot overflow.
///
/// If the return type is of type double, this routine
/// returns a scaling alpha such that y := (alpha * y) + t * (alpha * x)
/// cannot overflow.
///
/// If the return type is of type int, this routine
/// returns a scaling alpha such that y := (2^alpha * y) + t * (2^alpha * x)
/// cannot overflow.
///
/// Without checks, this routine assumes 0 <= t, x, y <= Omega.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] t
///         Scalar. 0 <= t <= Omega.
///
/// @param[in] x
///         Scalar. 0 <= x <= Omega.
///
/// @param[in] y
///         Scalar. 0 <= y <= Omega.
///
/// @return The scaling factor alpha.
///
scaling_t protect_update(double t, double x, double y);



///
/// @brief Computes scaling s.t. the update y:=y+t1*x1+t2*x2 cannot overflow.
///
/// If the return type is of type double, this routine
/// returns a scaling alpha such that y := (alpha * y) + t * (alpha * x)
/// cannot overflow.
///
/// If the return type is of type int, this routine
/// returns a scaling alpha such that y := (2^alpha * y) + t * (2^alpha * x)
/// cannot overflow.
///
/// Without checks, this routine assumes 0 <= t, x, y <= Omega.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] t1
///         Scalar. 0 <= t1 <= Omega.
///
/// @param[in] x1
///         Scalar. 0 <= x1 <= Omega.
///
/// @param[in] t2
///         Scalar. 0 <= t2 <= Omega.
///
/// @param[in] x2
///         Scalar. 0 <= x2 <= Omega.
///
/// @param[in] y
///         Scalar. 0 <= y <= Omega.
///
/// @return The scaling factor alpha.
///
scaling_t protect_double_update(double t1, double x1, double t2, double x2,
    double y);


///
/// @brief Computes scaling such that the update Y(:,i) := Y(:,i) + T X(:,i)
/// cannot overflow.
///
/// This routine wraps multiple calls to protect_update().
///
/// Without checks, this routine assumes that all norms satisfy 
/// 0 <= norm <= Omega.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] Xnorms
///         Array of length num_rhs. The i-th entry contains an upper bound
///         for X(:,i). 0 <= Xnorms[i] <= Omega.
///
/// @param[in] num_rhs
///         Number of right-hand sides. num_rhs >= 0.
///
/// @param[in] tnorm
///         Scalar, upper bounds of T. 0 <= tnorm <= Omega.
///
/// @param[in] Ynorms
///        Array of length num_rhs. The i-th entry contains an upper bound
///        for Y(:,i). 0 <= Ynorms[i] <= Omega.
///
/// @param[out] scales
///         Array of length num_rhs. The i-th entry holds a scaling factor
///         to survive Y(:,i) + T X(:,i).
///
/// @return Flag that indicates if rescaling is necessary (status == RESCALE)
///         or not (status == NO_RESCALE).
///
int protect_multi_rhs_update_real(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    scaling_t *restrict const scales);

int protect_multi_rhs_update_cmplx(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    scaling_t *restrict const scales);


///
///                         a + ib
/// @brief Solves p + iq = -------- robustly.
///                         c + id
///
/// If the type of scale is double, the routine solves (scale*(a + ib))/(c +id)
/// whereas, if the type of scale is int, the routine solves
/// (2^scale * (a + ib)) / (c + id) such that no overflow occurs.
///
/// @param[in] smin
///         Desired lower bound on (t - lambda).
///
/// @param[in] a
///         Real scalar a.
///
/// @param[in] b
///         Real scalar b representing the imaginary part.
///
/// @param[in] c
///         Real scalar c.
///
/// @param[in] d
///         Real scalar d representing the imaginary part.
///
/// @param[out] x_re
///         The real part of the solution.
///
/// @param[out] x_im
///         The imaginary part of the solution.
///
int robust_cmplx_div(double smin, double a, double b, double c, double d,
    double *restrict x_re, double *restrict x_im, scaling_t *scale);


///
/// @brief Solves (t - lambda) * ? = x robustly.
///
/// If the type of scale is double, the routine solves (scale * x) / (t - lambda)
/// whereas, if the type of scale is int, the routine solves
/// (2^scale * x) / (t - lambda) such that no overflow occurs.
///
/// @param[in] smin
///         Desired lower bound on (t - lambda).
///
/// @param[in] t
///         Real scalar t.
///
/// @param[in] lambda
///         Real scalar lambda.
///
/// @param[in, out] x
///         On entry, the scale rhs. On exit, the real solution x in
///         (scale * x) / (t - lambda) or in (2^scale * x) / (t - lambda).
///
/// @param[out] scale
///         Scalar scaling factor of x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if (t - lambda)
///         was perturbed to make it greater than smin.
///
int solve_1x1_real_system(
    double smin, double t, double lambda, double *x, scaling_t *scale);


///
/// @brief Solves (t - lambda_re - lambda_im) * ? = x_re + i * x_im robustly.
///
/// If the type of scale is double, the routine solves (scale * x) / (t - lambda)
/// whereas, if the type of scale is int, the routine solves
/// (2^scale * x) / (t - lambda) such that no overflow occurs. The complex
/// division is executed in real arithmetic.
///
/// @param[in] smin
///         Desired lower bound on (t - lambda_re - lambda_im).
///
/// @param[in] t
///         Real scalar t.
///
/// @param[in] lambda_re
///         Real part of the scalar complex eigenvalue.
///
/// @param[in] lambda_im
///         Imaginary part of the scalar complex eigenvalue.
///
/// @param[in, out] x_re
///         On entry, the real part of the right-hand side. On exit, the real
///         part of the solution.
///
/// @param[in, out] x_im
///         On entry, the imaginary part of the right-hand side. On exit, the
///         imaginary part of the solution.
///
/// @param[out] scale
///         Joint scalar scaling factor for the real and imaginary part of
///         the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if (t - lambda)
///         was perturbed to make it greater than smin.
///
int solve_1x1_cmplx_system(
    double smin,
    double t,
    double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t *scale);


///
/// @brief Solves a real-valued 2-by-2 system robustly.
///
/// Solves the real-valued system
///        [ t11-lambda  t12        ] * [ x1 ] = [ b1 ]
///        [ t21         t22-lambda ]   [ x2 ]   [ b2 ]
/// such that if cannot overflow.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] smin
///         Desired lower bound on the singular values of (T - lambda * I).
///
/// @param[in] T
///         Real 2-by-2 matrix T.
///
/// @param[in] ldT
///         The leading dimension of T. ldT >= 2.
///
/// @param[in] lambda
///         Real eigenvalue.
///
/// @param[in, out] b
///         Real vector of length 2. On entry, the right-hand side.
///         On exit, the solution.
///
/// @param[out] scale
///         Scalar scaling factor of the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if the singular
///         values of (T - lambda * I) were smaller than smin and perturbed.
///
int solve_2x2_real_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, scaling_t *restrict const scale);


///
/// @brief Solves a complex-valued 2-by-2 system robustly.
///
/// Let lambda := lambda_re + i * lambda_im. Solves the complex-valued system
/// [ t11-lambda_re   t12        ] * [ x1 ] = scale * ([ b_re1 ] + i * [b_im1])
/// [ t21             t22-lambda ]   [ x2 ]           ([ b_re2 ]       [b_im2])
/// such that it cannot overflow. The solution x1 and x2 is complex-valued.
///
/// Credits: Carl Christian Kjelgaard Mikkelsen.
///
/// @param[in] smin
///         Desired lower bound on the singular values of (T - lambda * I).
///
/// @param[in] T
///         Real 2-by-2 matrix T.
///
/// @param[in] ldT
///         The leading dimension of T. ldT >= 2.
///
/// @param[in] lambda_re
///         Real part of the eigenvalue.
///
/// @param[in] lambda_im
///         Imaginary part of the eigenvalue.
///
/// @param[in, out] b_re
///         Vector of length 2. On entry, the real part of the right-hand side.
///         On exit, the real part of the solution.
///
/// @param[in, out] b
///         Vector of length 2. On entry, the imaginary part of the right-hand
///         side. On exit, the imaginary part of the solution.
///
/// @param[out] scale
///         Scalar scaling factor of the solution x.
///
/// @return Error flag. Set to 0 if no error occurred. Set to 1 if the singular
///         values of (T - lambda * I) were smaller than smin and perturbed.
///
int solve_2x2_cmplx_system(
    double smin,
    const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    scaling_t *restrict const scale);

#endif
