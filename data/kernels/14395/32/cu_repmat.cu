#include "includes.h"
__global__ void cu_repmat(const float *a, float* dst, const int rowsa, const int colsa, const int rowsdst, const int colsdst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int c2 = tid % colsdst;
int r2 = tid / colsdst;
int ra = r2 % rowsa;
int ca = c2 % colsa;
dst[tid] = a[ra * colsa + ca];
tid += stride;
}
}