#include "includes.h"



__global__ void kernelUpdateParticle(float *positions,float *velocities,float *pBests,float *gBest,float r1,float r2)
{
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i>=NUM_OF_PARTICLES*NUM_OF_DIMENSIONS)
return;

float rp=r1;
float rg=r2;

velocities[i]=OMEGA*velocities[i]+c1*rp*(pBests[i]-positions[i])+c2*rg*(gBest[i%NUM_OF_DIMENSIONS]-positions[i]);
positions[i]+=velocities[i];
}