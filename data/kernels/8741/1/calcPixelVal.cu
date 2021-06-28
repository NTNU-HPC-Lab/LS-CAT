#include "includes.h"
__global__ void calcPixelVal(float *g_idata, float* constant, float* min)
{
unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;

if(i<LENA_SIZE)g_idata[i]=(g_idata[i]-(*min))*(*constant);

}