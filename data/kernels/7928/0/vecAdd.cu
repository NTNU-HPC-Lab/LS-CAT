#include "includes.h"
__global__ void vecAdd(float *in1,float *in2,float *out,int len)
{
// variable declarations
int i=blockIdx.x * blockDim.x + threadIdx.x;
// code
if(i < len)
{
out[i]=in1[i]+in2[i];
}
}