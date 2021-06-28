#include "includes.h"



__device__ float fitness_function(float x[])
{
float y,yp;
float res=0;
float y1=1+(x[0]-1)/4;
float yn=1+(x[NUM_OF_DIMENSIONS-1]-1)/4;

res+=pow(sin(phi*y1),2)+pow(yn-1,2);

for(int i=0;i<NUM_OF_DIMENSIONS-1;i++)
{
y=1+(x[i]-1)/4;
yp=1+(x[i+1]-1)/4;
res+=pow(y-1,2)*(1+10*pow(sin(phi*yp),2));
}

return res;
}
__global__ void kernelUpdatePBest(float *positions,float *pBests,float *gBest)
{
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i>=NUM_OF_PARTICLES*NUM_OF_DIMENSIONS||i%NUM_OF_DIMENSIONS!=0)
return;

float tempParticle1[NUM_OF_DIMENSIONS];
float tempParticle2[NUM_OF_DIMENSIONS];

for(int j=0;j<NUM_OF_DIMENSIONS;j++)
{
tempParticle1[j]=positions[i+j];
tempParticle2[j]=pBests[i+j];
}

if(fitness_function(tempParticle1)<fitness_function(tempParticle2))
{
for(int j=0;j<NUM_OF_DIMENSIONS;j++)
pBests[i+j]=tempParticle1[j];

if(fitness_function(tempParticle1)<fitness_function(gBest))
{
for(int j=0;j<NUM_OF_DIMENSIONS;j++)
atomicExch(gBest+j,tempParticle1[j]);
}
}
}