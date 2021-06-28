#include "includes.h"
__global__ void KerComputeVelMod(unsigned n,const float4 *vel,float *velmod)
{
unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n){
const float4 r=vel[p];
velmod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
}
}