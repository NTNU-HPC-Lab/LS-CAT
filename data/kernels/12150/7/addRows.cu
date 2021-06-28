#include "includes.h"
__global__ void addRows(double *matrix, int *d_i){
int i=*d_i;
int n=blockDim.x+i;
int id= n*(blockIdx.x+i+1) + threadIdx.x+i;
__shared__ double multiplier;

if(threadIdx.x==0){
multiplier=matrix[n*(blockIdx.x+1+i)+i]/matrix[n*i+i];
}
__syncthreads();

matrix[id]-=matrix[n*i+threadIdx.x+i]*multiplier;
}