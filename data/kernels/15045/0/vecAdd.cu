#include "includes.h"
/**********************************************************
* @author  Pulkit Verma
* @email   technopreneur[dot]pulkit[at]gmail[dot]com
**********************************************************/

// The program takes two equal size vectors as input and outputs their vector sum


__global__ void vecAdd(float *in1, float *in2, float *out, int len)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i<len)
out[i]=in1[i]+in2[i];

return;
}