#include "includes.h"
__global__ void cu_getRange(const float *src, float* dst, const int xstart, const int xend, const int ystart, const int yend, const int colssrc, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int colsdst = xend - xstart + 1;
while(tid < n){
int cdst = tid % colsdst;
int rdst = tid / colsdst;
int rsrc = rdst + ystart;
int csrc = cdst + xstart;
dst[tid] = src[rsrc * colssrc + csrc];
tid += stride;
}
}