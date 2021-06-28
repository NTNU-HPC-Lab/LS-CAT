#include "includes.h"
__global__ void set_carr(float br, float bi, float * c, int N)
{
int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;
int idc=idx*2;
c[idc]=br;c[idc+1]=bi;
}