#include "includes.h"
__global__ void kMartixSubstractMatrix(const int nThreads, const float *m1, const float *m2, float *output) {
/*  Computes the (elementwise) difference between two arrays
Inputs:
m1: array
m2: array
output: array,the results of the computation are to be stored here
*/

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = m1[i] - m2[i];
}
}