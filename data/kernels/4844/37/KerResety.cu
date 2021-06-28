#include "includes.h"
__global__ void KerResety(unsigned n,unsigned ini,float3 *v)
{
unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n)v[p+ini].y=0;
}