#include "includes.h"
__global__ void dotProduct_CUDA(double *sum, long size, double *vector1, double *vector2){
long idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
if(idx < size){
//printf("Before idx%d : %lf\n",idx,sum[idx]);
sum[idx] = (vector2[idx]) * (vector1[idx]);
//printf("Vector1 %lf\n",*(vector1+idx));
//printf("Vector2 %lf\n",vector2[idx]);
//printf("After idx%d : %lf\n",idx,sum[idx]);
}
}