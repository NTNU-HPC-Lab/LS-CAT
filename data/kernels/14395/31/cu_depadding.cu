#include "includes.h"
__global__ void cu_depadding(const float* src, float* dst, const int rows1, const int cols1, const int cols2, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int pad = (cols1 - cols2) / 2;
int c2 = tid % cols2;
int r2 = tid / cols2;
int r1 = r2 + pad;
int c1 = c2 + pad;
dst[tid] = src[r1 * cols1 + c1];
tid += stride;
}
}