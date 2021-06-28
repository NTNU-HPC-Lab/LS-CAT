#include "includes.h"
__global__ void ARR_ADDC(float* result, float* in1, float* in2, int N)
{
int index = blockDim.x * blockIdx.x + threadIdx.x;
if (index < N)
{
result[index] = in1[index] + in2[index];
}
}