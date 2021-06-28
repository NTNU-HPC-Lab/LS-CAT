#include "includes.h"
__global__ void _mat_sum_col(float *m, float *target,int nrow, int ncol){
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < ncol){
float sum = 0;
for(int i = 0; i < nrow; i++){
sum += m[i*ncol+tid];
}
target[tid] = sum;
}
}