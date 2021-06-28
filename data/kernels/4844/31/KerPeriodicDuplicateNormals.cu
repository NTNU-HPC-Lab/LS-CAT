#include "includes.h"
__global__ void KerPeriodicDuplicateNormals(unsigned n,unsigned pini,const unsigned *listp,float3 *normals,float3 *motionvel)
{
const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<n){
const unsigned pnew=p+pini;
const unsigned rp=listp[p];
const unsigned pcopy=(rp&0x7FFFFFFF);
normals[pnew]=normals[pcopy];
if(motionvel)motionvel[pnew]=motionvel[pcopy];
}
}