#include "includes.h"
__global__ void Ecalc2(float* out, const float* label)
{
int i = blockDim.x*blockIdx.x + threadIdx.x; //10 * Data.count
out[i] = label[i] - out[i];
}