#include "includes.h"
__global__ void reduceVector(float *v1, float *v2, float *res){


int index = blockIdx.x * blockDim.x + threadIdx.x;
int index2;

for (int i = blockDim.x/2; i>=1; i=i/2){

if(threadIdx.x < i){
index2 = index + i;
v1[index] += v1[index2];
}
__syncthreads();

}
if(threadIdx.x==0)
res[blockIdx.x] = v1[index];

}