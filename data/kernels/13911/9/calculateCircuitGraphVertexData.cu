#include "includes.h"
__global__ void calculateCircuitGraphVertexData( unsigned int * D,unsigned int * C,unsigned int ecount){

unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
if( tid <ecount)
{
unsigned int c=D[tid];
atomicExch(C+c,1);
}
}