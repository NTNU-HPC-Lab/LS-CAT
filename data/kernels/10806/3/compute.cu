#include "includes.h"
__global__ void compute(int *v1,int *v2, int *v3, int N){
//blockIdx.x (0-2) threadIdx.x (0-99)
if(blockIdx.x==2){
v3[(N*blockIdx.x) + threadIdx.x] = v1[((blockIdx.x-2)*N)+threadIdx.x]*v2[((blockIdx.x-1)*N)+threadIdx.x] -
v1[((blockIdx.x-1)*N)+threadIdx.x]*v2[((blockIdx.x-2)*N)+threadIdx.x];
}else if(blockIdx.x==1){
v3[(N*blockIdx.x) + threadIdx.x] = v1[((blockIdx.x+1)*N)+threadIdx.x]*v2[((blockIdx.x-1)*N)+threadIdx.x] -
v1[(N*(blockIdx.x-1))+threadIdx.x]*v2[((blockIdx.x+1)*N)+threadIdx.x];
}else{
v3[(N*blockIdx.x) + threadIdx.x] = v1[((blockIdx.x+1)*N)+threadIdx.x]*v2[((blockIdx.x+2)*N)+threadIdx.x] -
v2[((blockIdx.x+1)*N)+threadIdx.x]*v1[((blockIdx.x+2)*N)+threadIdx.x];
}
}