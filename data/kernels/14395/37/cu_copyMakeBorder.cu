#include "includes.h"
__global__ void cu_copyMakeBorder(const float *src, float* dst, const int rowssrc, const int colssrc, const int up, const int down, const int left, const int right, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int colsdst = colssrc + left + right;
while(tid < n){
int csrc = tid % colssrc;
int rsrc = tid / colssrc;
int rdst = up + rsrc;
int cdst = left + csrc;
dst[rdst * colsdst + cdst] = src[tid];
tid += stride;
}
}