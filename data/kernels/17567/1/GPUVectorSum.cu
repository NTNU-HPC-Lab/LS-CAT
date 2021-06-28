#include "includes.h"
__global__ void GPUVectorSum(int * a, int * b, int * c, int VECTOR_QNT) {
int n = VECTOR_QNT;
int idx = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = idx; i < n; i += blockDim.x * gridDim.x)
{
c[i] = a[i] + b[i];
}
}