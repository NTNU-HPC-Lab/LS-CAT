#include "includes.h"
__global__ void cu_pooling_overlap_max(const float* src, float* dst, float *loc, const int rowssrc, const int colssrc, const int rowsdst, const int colsdst, const int sizex, const int sizey, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int cdst = tid % colsdst;
int rdst = tid / colsdst;
int rsrc = rdst;
int csrc = cdst;
int xend = (csrc + sizex - 1);
int yend = (rsrc + sizey - 1);
loc[tid] = (float)(rsrc * colssrc + csrc);
for(int i = rsrc; i <= yend; ++i){
for(int j = csrc; j <= xend; ++j){
if(src[i * colssrc + j] > dst[tid]){
dst[tid] = src[i * colssrc + j];
loc[tid] = (float)(i * colssrc + j);
}
}
}
tid += stride;
}
}