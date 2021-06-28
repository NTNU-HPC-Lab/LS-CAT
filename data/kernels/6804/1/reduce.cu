#include "includes.h"
__global__ void reduce(double *a,double *z, int sizeOut){
int tid = blockDim.x*blockIdx.x + threadIdx.x;
if(tid > N/2)return;

extern __shared__ double subTotals[];
subTotals[threadIdx.x]=(a[tid*2]+a[tid*2+1])/2;//sum every two values using all threads
__syncthreads();
int level=2;
while ((blockDim.x/level) >= sizeOut){//keep halving values until sizeout remains
if(threadIdx.x % level==0){//use half threads every iteration
subTotals[threadIdx.x]=(subTotals[threadIdx.x]+subTotals[threadIdx.x+(level/2)])/2;
}
__syncthreads();//we have to sync threads every time here :(
level = level * 2;
}
level = level /2;
if(threadIdx.x % level==0){
z[tid/level] = subTotals[threadIdx.x];
}
}