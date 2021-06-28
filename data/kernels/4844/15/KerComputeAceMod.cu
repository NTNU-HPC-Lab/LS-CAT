#include "includes.h"
__global__ void KerComputeAceMod(unsigned n,const float3 *ace,float *acemod)
{
unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n){
const float3 r=ace[p];
acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
}
}