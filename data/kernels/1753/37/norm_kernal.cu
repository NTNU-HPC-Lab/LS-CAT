#include "includes.h"
__global__ void norm_kernal(float * data, float mean, float var, int totaltc)
{

const uint index = threadIdx.x + (blockIdx.x + gridDim.x*blockIdx.y)*MAX_THREADS;

if(index < totaltc){
data[index] = (data[index] - mean)/var;
}

}