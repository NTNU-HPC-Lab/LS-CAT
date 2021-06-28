#include "includes.h"
__global__ void add_vectors(float *ad, float *bd, int N)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < N)
ad[index] += bd[index];                             //adding values in GPU memory
}