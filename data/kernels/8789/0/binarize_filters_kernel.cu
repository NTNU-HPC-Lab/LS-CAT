#include "includes.h"
__global__ void binarize_filters_kernel(float *filters, int n, int size, float *binary)
{
int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (f >= n) return;
int i = 0;
float mean = 0;
for(i = 0; i < size; ++i){
mean += abs(filters[f*size + i]);
}
mean = mean / size;
for(i = 0; i < size; ++i){
binary[f*size + i] = (filters[f*size + i] > 0) ? mean : -mean;
}
}