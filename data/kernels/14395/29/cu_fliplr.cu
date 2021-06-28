#include "includes.h"
__global__ void cu_fliplr(const float* src, float* dst, const int rows, const int cols, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while(tid < n){
int c = tid % cols;
int r = tid / cols;
dst[tid] = src[(cols - c - 1) + r * cols];
tid += stride;
}
}