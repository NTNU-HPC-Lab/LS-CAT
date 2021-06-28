#include "includes.h"
__global__ void gpu_transpose(const float* src, float* dst, int colssrc, int colsdst, int n) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while (tid < n) {
int cdst = tid % colsdst;
int rdst = tid / colsdst;
int rsrc = cdst;
int csrc = rdst;
dst[tid] = src[rsrc * colssrc + csrc];
tid += stride;
}
}