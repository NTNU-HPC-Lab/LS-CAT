#include "includes.h"
__global__ void vecDiv(float* a,float* b,float* c,const int N){
const int i = blockIdx.x*blockDim.x + threadIdx.x;
if(i<N)
c[i] = __fdividef(a[i],b[i]);
//c[i] = a[i]/b[i];
}