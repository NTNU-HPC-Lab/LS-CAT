#include "includes.h"
__global__ void subtract_kernal(float* data, float f, const int totaltc)
{

int idx = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*MAX_THREADS;

if(idx < totaltc){
data[idx] = data[idx] - f;
}
}