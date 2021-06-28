#include "includes.h"
__global__ void _mat_sum_row(float *m, float *target,int nrow, int ncol){
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < nrow){
float sum = 0;
for(int i = 0; i < ncol; i++){
sum += m[tid*ncol+i];
}
target[tid] = sum;
}
}