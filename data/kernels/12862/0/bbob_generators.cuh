#include <cuda.h>
#include <math_functions.h>

const int MAX_DIMENSIONS = 40;
#define coco_pi 3.14159265358979323846
#define coco_two_pi 2.0 * 3.14159265358979323846
#define d_OMEGA 0.64
#define d_phi 1.4

__device__ double coco_double_round(const double number) {
	return floor(number + 0.5);
}

__device__ double coco_double_max(const double a, const double b) {
	if (a >= b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ void bbob2009_unif(double *r, size_t N, long inseed) {
	/* generates N uniform numbers with starting seed */
	long aktseed;
	long tmp;
	long rgrand[32];
	long aktrand;
	long i;

	if (inseed < 0)
		inseed = -inseed;
	if (inseed < 1)
		inseed = 1;
	aktseed = inseed;
	for (i = 39; i >= 0; i--) {
		tmp = (int)floor((double)aktseed / (double)127773);
		aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
		if (aktseed < 0)
			aktseed = aktseed + 2147483647;
		if (i < 32)
			rgrand[i] = aktseed;
	}
	aktrand = rgrand[0];
	for (i = 0; i < N; i++) {
		tmp = (int)floor((double)aktseed / (double)127773);
		aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
		if (aktseed < 0)
			aktseed = aktseed + 2147483647;
		tmp = (int)floor((double)aktrand / (double)67108865);
		aktrand = rgrand[tmp];
		rgrand[tmp] = aktseed;
		r[i] = (double)aktrand / 2.147483647e9;
		if (r[i] == 0.) {
			r[i] = 1e-99;
		}
	}
	return;
}

__device__ void bbob2009_gauss(double *g, const size_t N, const long seed) {
	size_t i;
	double uniftmp[6000];
	bbob2009_unif(uniftmp, 2 * N, seed);

	for (i = 0; i < N; i++) {
		g[i] = sqrt(-2 * log(uniftmp[i])) * cos(2 * coco_pi * uniftmp[N + i]);
		if (g[i] == 0.)
			g[i] = 1e-99;
	}
}

__device__ double bbob2009_fmin(double a, double b) {
	return (a < b) ? a : b;
}

__device__ double bbob2009_fmax(double a, double b) {
	return (a > b) ? a : b;
}

__device__ double bbob2009_round(double x) {
	return floor(x + 0.5);
}

__device__ double bbob2009_compute_fopt(const int function, const int instance) {
	long rseed, rrseed;
	double gval, gval2;

	if (function == 4)
		rseed = 3;
	else if (function == 18)
		rseed = 17;
	else if (function == 101 || function == 102 || function == 103 || function == 107
		|| function == 108 || function == 109)
		rseed = 1;
	else if (function == 104 || function == 105 || function == 106 || function == 110
		|| function == 111 || function == 112)
		rseed = 8;
	else if (function == 113 || function == 114 || function == 115)
		rseed = 7;
	else if (function == 116 || function == 117 || function == 118)
		rseed = 10;
	else if (function == 119 || function == 120 || function == 121)
		rseed = 14;
	else if (function == 122 || function == 123 || function == 124)
		rseed = 17;
	else if (function == 125 || function == 126 || function == 127)
		rseed = 19;
	else if (function == 128 || function == 129 || function == 130)
		rseed = 21;
	else
		rseed = (long)function;

	rrseed = rseed + (long)(10000 * instance);
	bbob2009_gauss(&gval, 1, rrseed);
	bbob2009_gauss(&gval2, 1, rrseed + 1);
	return bbob2009_fmin(1000., bbob2009_fmax(-1000., bbob2009_round(100. * 100. * gval / gval2) / 100.));
}

__device__ void bbob2009_compute_xopt(double *xopt, const long seed, const size_t DIM) {
	long i;
	bbob2009_unif(xopt, DIM, seed);
	for (i = 0; i < DIM; i++) {
		xopt[i] = 8 * floor(1e4 * xopt[i]) / 1e4 - 4;
		if (xopt[i] == 0.0)
			xopt[i] = -1e-5;
	}
}

__device__ void transform_obj_oscillate(double* y, int number_of_objectives)
{
	const double factor = 0.1;
	size_t i;

	for (i = 0; i < number_of_objectives; i++) {
		if (y[i] != 0) {
			double log_y;
			log_y = log(fabs(y[i])) / factor;
			if (y[i] > 0) {
				y[i] = pow(exp(log_y + 0.49 * (sin(log_y) + sin(0.79 * log_y))), factor);
			}
			else {
				y[i] = -pow(exp(log_y + 0.49 * (sin(0.55 * log_y) + sin(0.31 * log_y))), factor);
			}
		}
	}
}

__device__ void transform_obj_power(double* y, int number_of_objectives)
{
	const double exponent = 0.9;
	int i = 0;

    for (i = 0; i < number_of_objectives; i++) {
		y[i] = pow(y[i], exponent);
	}
}

__device__ void transform_obj_shift(double* y, int number_of_objectives, double offset)
{
	int i = 0;

	for (i = 0; i < number_of_objectives; i++) {
		y[i] += offset;
	}
}

__device__ void transform_vars_affine(double* x, int number_of_variables, double* M, double* b)
{
	int i, j;
	double temp[MAX_DIMENSIONS];

    for (i = 0; i < number_of_variables; ++i) {
		/* data->M has problem->number_of_variables columns and inner_problem->number_of_variables rows. */
		const double *current_row = M + i * number_of_variables;
		temp[i] = b[i];
		for (j = 0; j < number_of_variables; ++j) {
			temp[i] += x[j] * current_row[j];
		}
	}

	for (i = 0; i < number_of_variables; ++i) {
		x[i] = temp[i];
	}
}

__device__ void transform_vars_shift(double* x, int number_of_variables, double* offset)
{
    int i;

	for (i = 0; i < number_of_variables; ++i) {
		x[i] = x[i] - offset[i];
	}
}

__device__ void bbob2009_reshape(const size_t DIM, double B[40][40], double *vector) {
	size_t i, j;
	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			B[i][j] = vector[j * DIM + i];
		}
	}
}

__device__ void bbob2009_compute_rotation(const size_t DIM, double B[40][40], const long seed) {
	double prod;
	double gvect[2000];
	long i, j, k;

	bbob2009_gauss(gvect, DIM * DIM, seed);
	bbob2009_reshape(DIM, B, gvect);

	for (i = 0; i < DIM; i++) {
		for (j = 0; j < i; j++) {
			prod = 0;
			for (k = 0; k < DIM; k++)
				prod += B[k][i] * B[k][j];
			for (k = 0; k < DIM; k++)
				B[k][i] -= prod * B[k][j];
		}
		prod = 0;
		for (k = 0; k < DIM; k++)
			prod += B[k][i] * B[k][i];
		for (k = 0; k < DIM; k++)
			B[k][i] /= sqrt(prod);
	}
}

__device__ void bbob2009_copy_rotation_matrix(double rot[MAX_DIMENSIONS][MAX_DIMENSIONS], double *M, double *b, const size_t DIM) {
	size_t row, column;
	double *current_row;

	for (row = 0; row < DIM; ++row) {
		current_row = M + row * DIM;
		for (column = 0; column < DIM; ++column) {
			current_row[column] = rot[row][column];
		}
		b[row] = 0.0;
	}
}

__device__ void transform_vars_asymmetric(double *x, int number_of_variables, double beta)
{
	int i;
	double exponent;

    for (i = 0; i < number_of_variables; ++i) {
		if (x[i] > 0.0) {
			exponent = 1.0
				+ (beta * (double)(long)i) / ((double)(long)number_of_variables - 1.0) * sqrt(x[i]);
			x[i] = pow(x[i], exponent);
		}
		else {
			x[i] = x[i];
		}
	}
}

__device__ void transform_vars_brs(double *x, int number_of_variables)
{
    for (int i = 0; i < number_of_variables; ++i) {
		/* Function documentation says we should compute 10^(0.5 *
		* (i-1)/(D-1)). Instead we compute the equivalent
		* sqrt(10)^((i-1)/(D-1)) just like the legacy code.
		*/
		double factor = pow(sqrt(10.0), (double)(long)i / ((double)(long)number_of_variables - 1.0));
		/* Documentation specifies odd indices and starts indexing
		* from 1, we use all even indices since C starts indexing
		* with 0.
		*/
		if (x[i] > 0.0 && i % 2 == 0) {
			factor *= 10.0;
		}
		x[i] = factor * x[i];
	}
}

__device__ void transform_vars_oscillate(double *x, int number_of_variables)
{
	const double alpha = 0.1;
	double tmp, base;

    for (int i = 0; i < number_of_variables; ++i) {
		if (x[i] > 0.0) {
			tmp = log(x[i]) / alpha;
			base = exp(tmp + 0.49 * (sin(tmp) + sin(0.79 * tmp)));
			x[i] = pow(base, alpha);
		}
		else if (x[i] < 0.0) {
			tmp = log(-x[i]) / alpha;
			base = exp(tmp + 0.49 * (sin(0.55 * tmp) + sin(0.31 * tmp)));
			x[i] = -pow(base, alpha);
		}
		else {
			x[i] = 0.0;
		}
	}
}

__device__ void transform_obj_penalize(double *y, int number_of_objectives, double factor)
{
    for (int i = 0; i < number_of_objectives; ++i) {
		y[i] += factor * 0.5;
	}
}

__device__ void transform_vars_conditioning(double *x, int number_of_variables, double alpha)
{
    for (int i = 0; i < number_of_variables; ++i) {
		/* OME: We could precalculate the scaling coefficients if we
		* really wanted to.
		*/
		x[i] = pow(alpha, 0.5 * (double)(long)i / ((double)(long)number_of_variables - 1.0))
			* x[i];
	}
}

__device__ void transform_vars_scale(double *x, int number_of_variables, double factor)
{
    for (int i = 0; i < number_of_variables; ++i) {
		x[i] = factor * x[i];
	}
}

__device__ void transform_vars_x_hat(double *x, int number_of_variables, double seed)
{
	double tmp[MAX_DIMENSIONS];
    bbob2009_unif(tmp, number_of_variables, seed);

	for (int i = 0; i < number_of_variables; ++i) {
		if (tmp[i] - 0.5 < 0.0) {
			x[i] = -x[i];
		}
		else {
			x[i] = x[i];
		}
	}
}

__device__ void transform_vars_z_hat(double *x, int number_of_variables, double* xopt)
{
	double z[MAX_DIMENSIONS];

    z[0] = x[0];

	for (int i = 1; i < number_of_variables; ++i) {
		z[i] = x[i] + 0.25 * (x[i - 1] - 2.0 * fabs(xopt[i - 1]));
	}

	for (int i = 0; i < number_of_variables; ++i)
	{
	    x[i] = z[i];
	}
}

__device__ double coco_double_min(const double a, const double b) {
	if (a <= b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ void clamp(double* vector, int length, double min, double max)
{
    for(int i = 0; i < length; i++)
    {
        vector[i] = vector[i] < min ? min : (vector[i] > max ? max : vector[i]);
    }
}

__device__ void vector_between(double* from, double* to, int dimensions, double* result)
{
    for(int i = 0; i < dimensions; i++)
    {
        result[i] = to[i] - from[i];
    }
}