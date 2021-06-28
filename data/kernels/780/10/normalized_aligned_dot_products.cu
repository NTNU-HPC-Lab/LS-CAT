#include "includes.h"
__global__ void normalized_aligned_dot_products(const double* A, const double divisor, const unsigned int m, const unsigned int n, double* QT)
{
int a = blockIdx.x * blockDim.x + threadIdx.x;
if (a < n) {
QT[a] = A[a + m - 1] / divisor;
}
}