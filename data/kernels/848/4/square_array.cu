#include "includes.h"
__global__ void square_array(float *ad, int N)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < N)
ad[index] *= ad[index];                             //adding values in GPU memory
}