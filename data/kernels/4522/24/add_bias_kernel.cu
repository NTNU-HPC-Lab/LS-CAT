#include "includes.h"
__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= n*size*batch) return;
int i = index % size;
index /= size;
int j = index % n;
index /= n;
int k = index;

output[(k*n+j)*size + i] += biases[j];
}