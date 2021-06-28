#include "includes.h"
__global__ void cu_pooling_mean(const float* src, float* dst, float *loc, const int rowssrc, const int colssrc, const int rowsdst, const int colsdst, const int stridex, const int stridey, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int cdst = tid % colsdst;
int rdst = tid / colsdst;
int rsrc = rdst * stridey;
int csrc = cdst * stridex;
int xend = (csrc + stridex - 1) > (colssrc - 1) ? (colssrc - 1) : (csrc + stridex - 1);
int yend = (rsrc + stridey - 1) > (rowssrc - 1) ? (rowssrc - 1) : (rsrc + stridey - 1);
loc[tid] = (float)(rsrc * colssrc + csrc);
for(int i = rsrc; i <= yend; ++i){
for(int j = csrc; j <= xend; ++j){
dst[tid] += __fdividef(src[i * colssrc + j], __fmul_rd(yend - rsrc + 1, xend - csrc + 1));
}
}
tid += stride;
}
}