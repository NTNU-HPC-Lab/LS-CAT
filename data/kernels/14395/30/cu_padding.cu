#include "includes.h"
__global__ void cu_padding(const float* src, float* dst, const int rows1, const int cols1, const int cols2, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int pad = (cols2 - cols1) / 2;
int c1 = tid % cols1;
int r1 = tid / cols1;
int r2 = r1 + pad;
int c2 = c1 + pad;
dst[r2 * cols2 + c2] = src[tid];
tid += stride;
}
}