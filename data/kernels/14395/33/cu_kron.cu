#include "includes.h"
__global__ void cu_kron(const float *a, const float* b, float* dst, const int rowsa, const int colsa, const int rowsdst, const int colsdst, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int colsb = colsdst / colsa;
int rowsb = rowsdst / rowsa;
while(tid < n){
int c2 = tid % colsdst;
int r2 = tid / colsdst;
int rb = r2 % rowsb;
int cb = c2 % colsb;
int ra = r2 / rowsb;
int ca = c2 / colsb;
dst[tid] = a[ra * colsa + ca] * b[rb * colsb + cb];
tid += stride;
}
}