#include "includes.h"
__global__ void solve(float* mat, float* b, float* x, int rows, int cols)
{
int n = blockIdx.x*threads1D + threadIdx.x;
if (n < rows) //Ensure bounds
x[n] = b[n] / mat[n * cols + n];
}