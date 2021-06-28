#include "includes.h"
__global__ void kMartixByMatrixElementwise(const int nThreads, const float *m1, const float *m2, float *output) {
/*  Computes the product of two arrays (elementwise multiplication).
Inputs:
m1: array
m2: array
output: array,the results of the multiplication are to be stored here
*/
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = m1[i] * m2[i];
}
}