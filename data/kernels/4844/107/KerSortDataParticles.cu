#include "includes.h"
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const double2 *a,const double *b,const float4 *c,double2 *a2,double *b2,float4 *c2)
{
const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Particle number.
if(p<n){
const unsigned oldpos=(p<pini? p: sortpart[p]);
a2[p]=a[oldpos];
b2[p]=b[oldpos];
c2[p]=c[oldpos];
}
}