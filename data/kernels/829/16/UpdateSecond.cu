#include "includes.h"
__global__ void UpdateSecond(float *WHAT , float *WITH , float AMOUNT , float *MULT)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
WHAT[idx] *=MULT[idx];
WHAT[idx] +=AMOUNT*WITH[idx];
MULT[idx] = 1.0f;
}