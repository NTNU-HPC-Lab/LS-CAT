#include "includes.h"
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{

int idx = threadIdx.x + blockIdx.x * blockDim.x;
int stride = gridDim.x * blockDim.x;

for(int i = idx; i < N; i += stride)
{
result[i] = a[i] + b[i];
}
}