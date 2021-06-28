#include "includes.h"
__global__ void set_arr(float b, float * c, int N)
{
int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;
c[idx]=b;
}