#include "includes.h"
__global__ void componentStepFive(unsigned int * Q,unsigned int length,unsigned  int * sprimtemp,unsigned int s){
unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
if(tid <length) {
if(Q[tid]==s){
atomicExch(sprimtemp,1);
//*sprime=*sprimtemp+1;
}
}
}