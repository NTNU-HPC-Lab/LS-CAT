#include "includes.h"
__global__ void divide(float *x, float* y ,float* out ,const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size)
{
out[index] = x[index]/y[index] ;
}
}