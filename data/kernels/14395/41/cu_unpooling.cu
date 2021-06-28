#include "includes.h"
__global__ void cu_unpooling(const float* src, const float* loc, float* dst, const int colsdst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int cdst = (int)(loc[tid]) % colsdst;
int rdst = (int)(loc[tid]) / colsdst;
dst[rdst * colsdst + cdst] = src[tid];
tid += stride;
}
}