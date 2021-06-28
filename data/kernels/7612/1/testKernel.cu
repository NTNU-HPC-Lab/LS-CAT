#include "includes.h"
__global__ void testKernel( float* g_idata, float* g_odata)
{
float result=1;
// read two values
float val1 = g_idata[0];
float val2 = g_idata[1];

// place loop/unrolled loop here to do a bunch of multiply add ops
// make sure you use results, so compiler does not optomize out
result = val2 + (result * val1);

g_odata[0] = result;
}