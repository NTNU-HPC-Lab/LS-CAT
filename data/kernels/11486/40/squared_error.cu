#include "includes.h"
__global__ void squared_error ( const float * ideal, float * actual, float * errors )
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
float diff = ideal[x] - actual[x];
errors[x] = __fmul_rz(diff,diff);
//printf("squared_error: %f, ideal: %f, actual: %f\n",errors[x],ideal[x],actual[x]);
}