#include "includes.h"

/*
* lanczos computes the smallest n_eigs eigenvalues for dev_L and the
* corresponding eigenvectors using the Lanczos algorithm.
*
* F: an array (n_patch by n_eigs) to store the eigenvectors
* Es: an array (1 by n_eigs) to store the eigenvalues
* dev_L: an array (n_patch by n_patch) representing the Laplacian matrix
* n_patch: the dimension of dev_L
*/
static double norm2(double *v, int length);

__global__ void divide_copy(double *dest, const double *src, int length, const double divisor)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
double factor = 1.0 / divisor;
while (tid < length) {
dest[tid] = src[tid] * factor;
tid += blockDim.x * gridDim.x;
}
}