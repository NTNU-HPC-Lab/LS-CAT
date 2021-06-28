#include "includes.h"
__global__ void do_sum_merge(int *datas, int n){
int tid=blockDim.x*threadIdx.y+threadIdx.x;
//int idx=blockIdx.x*blockDim.x+threadIdx.x;
//int idy=blockIdx.y*blockDim.y+threadIdx.y;
//int bid=gridDim.x*blockDim.x*idy+idx;
while(n>1){
if (tid< (1+n)/2 && n-1-tid!=tid){
datas[tid]+=datas[n-1-tid];
printf ("%d->%d->%d\n",n,tid,datas[tid]);
}
n/=2;
__syncthreads();
}
}