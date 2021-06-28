#include "includes.h"
__global__ void KerInOutUpdateVelrhopM1(unsigned n,const int *inoutpart ,const float4 *velrhop,float4 *velrhopm1)
{
const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
if(cp<n){
const unsigned p=inoutpart[cp];
velrhopm1[p]=velrhop[p];
}
}