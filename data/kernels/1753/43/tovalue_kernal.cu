#include "includes.h"
__global__ void tovalue_kernal(float* data, const float value, const int totaltc)
{

const uint idx = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*MAX_THREADS;

if(idx < totaltc){
data[idx] = value;
}

}