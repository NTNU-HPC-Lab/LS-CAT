#include "includes.h"
__global__ void logicalkernel(bool *A, bool *B, int *neighbours, int order ,int degree) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx <order){
for(int i=0 ; i <  degree ; i++){
int n = neighbours[idx*degree + i ];
for(int j = 0; j < order; j++){
B[idx * order+ j] = B[idx*order+j] || A[n*order+j];
}
}
}
}