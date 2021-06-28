#include "includes.h"
__global__ void mcfauto_kernal(const float* data1, float* data2, const int totaltc)
{
int idx = 2*(threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*MAX_THREADS);

if(idx < totaltc){
data2[idx] = sqrt(data1[idx] * data2[idx] + data1[idx + 1] * data2[idx + 1]);
data2[idx + 1] = 0;
}
}