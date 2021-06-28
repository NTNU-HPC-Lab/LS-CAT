#include "includes.h"
__global__ void KerInOutInterpolateTime(unsigned npt,double fxtime ,const float *vel0,const float *vel1,float *vel)
{
const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(p<npt){
const float v0=vel0[p];
vel[p]=float(fxtime*(vel1[p]-v0)+v0);
}
}