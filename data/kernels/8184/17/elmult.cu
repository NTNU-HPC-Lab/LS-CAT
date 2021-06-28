#include "includes.h"
__global__ void elmult(float * inA, float * inB, int length)
{
int idx = threadIdx.x + blockDim.x*blockIdx.x;
if (idx<length) inA[idx] *= inB[idx];
}