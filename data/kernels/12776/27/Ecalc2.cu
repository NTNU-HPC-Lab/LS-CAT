#include "includes.h"
__global__ void Ecalc2(float* out, const float* label)
{
int i = threadIdx.x;                         //4
//int j = blockDim.y*blockIdx.y + threadIdx.y; //Data.count

out[i] = label[i] - out[i];
}