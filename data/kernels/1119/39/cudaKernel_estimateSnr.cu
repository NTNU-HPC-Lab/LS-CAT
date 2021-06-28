#include "includes.h"
__global__ void cudaKernel_estimateSnr(const float* corrSum, const int* corrValidCount, const float* maxval, float* snrValue, const int size)

{
int idx = threadIdx.x + blockDim.x*blockIdx.x;

if (idx >= size) return;

float mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);

snrValue[idx] = maxval[idx] / mean;
}