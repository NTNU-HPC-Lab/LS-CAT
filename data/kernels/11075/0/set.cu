#include "includes.h"

#define imin(a,b) (a<b?a:b)
__global__ void set(double *dx,int N)
{
int tid=threadIdx.x+blockIdx.x*blockDim.x ;
if (tid<N)
dx[tid]=0.0;
}