#include "includes.h"
__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= N) return;
X[index*INCX] = ALPHA;
}