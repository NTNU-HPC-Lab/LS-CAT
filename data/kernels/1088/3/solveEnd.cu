#include "includes.h"
__global__ static void solveEnd ( double* data,  const double a, const double b, const double d, const double e,  const double omega_11, const double omega_12, const double omega_21, const double omega_22,  const int nx, const int nBatch )
{
// Matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

// Last two vectors
double newNx2;
double newNx1;

// Compute lambda = d^~ - transpose(g) * inverse(E) * d_hat
newNx2 = data[(nx - 2) * nBatch + globalIdx] - (e * data[globalIdx] + a * data[(nx - 4) * nBatch + globalIdx] + b * data[(nx - 3) * nBatch + globalIdx]);
newNx1 = data[(nx - 1) * nBatch + globalIdx] - (d * data[globalIdx] + e * data[nBatch + globalIdx] + a * data[(nx - 3) * nBatch + globalIdx]);

// Compute x^~ = omega * lambda
data[(nx - 2) * nBatch + globalIdx] = omega_11 * newNx2 + omega_12 * newNx1;
data[(nx - 1) * nBatch + globalIdx] = omega_21 * newNx2 + omega_22 * newNx1;
}