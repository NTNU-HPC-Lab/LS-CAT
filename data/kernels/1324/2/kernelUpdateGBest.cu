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
__global__ void kernelUpdateGBest(float *gBest,float *pBests)
{
float temp[NUM_OF_DIMENSIONS];
for(int i=0;i<10*NUM_OF_DIMENSIONS;i+=NUM_OF_DIMENSIONS)
{
for(int k=0;k<NUM_OF_DIMENSIONS;k++)
temp[k]=pBests[i+k];

if(fitness_function(temp)<fitness_function(gBest))
{
for(int k=0;k<NUM_OF_DIMENSIONS;k++)
gBest[k]=temp[k];
}
}
}