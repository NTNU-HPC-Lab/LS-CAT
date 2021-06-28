#include "includes.h"
__global__ void cuda_activateTanh(double* pA, int n)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < n) {
pA[id] = tanh(pA[id]);
}
}