#include "includes.h"
__global__ void vecDiv(float* a,float* b,float* c,const int N){
//const int i = blockIdx.x*blockDim.x + threadIdx.x;
const int i = gridDim.x*blockDim.x*blockIdx.y +  blockIdx.x*blockDim.x + threadIdx.x;
if(i<N)
c[i] = a[i]/b[i];
//c[i] = __fdividef(a[i],b[i]);  //faster, less-accurate divide
}