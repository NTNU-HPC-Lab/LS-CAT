#include "includes.h"
__global__ void constructCircuitGraphVertex(unsigned int * C,unsigned int * offset,unsigned int ecount, unsigned int * cv, unsigned int cvCount){
unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
if(tid < ecount){
if(C[tid]!=0){
cv[offset[tid]]=tid;
}
}
}