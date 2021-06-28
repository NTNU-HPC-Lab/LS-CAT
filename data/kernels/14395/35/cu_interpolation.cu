#include "includes.h"
__global__ void cu_interpolation(const float* src, float* dst, const int colssrc, const int colsdst, const int _stride, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int csrc = tid % colssrc;
int rsrc = tid / colssrc;
int rdst = rsrc * _stride;
int cdst = csrc * _stride;
dst[rdst * colsdst + cdst] = src[tid];
tid += stride;
}
}