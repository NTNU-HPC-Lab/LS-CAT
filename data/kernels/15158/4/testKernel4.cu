#include "includes.h"
__global__ void testKernel4(float *data1, float *data2)
{
float t = 0.0f;
float c = 0.0f;

//printf("d = %f\n", data1[NX*blockIdx.x + threadIdx.x]);

if(blockIdx.x > 0)
{
t += (data1[NX*(blockIdx.x-1)+threadIdx.x] - data1[NX*blockIdx.x + threadIdx.x]);
c += 1.0f;
}
if(blockIdx.x < NX-1)
{
t += (data1[NX*(blockIdx.x+1)+threadIdx.x] - data1[NX*blockIdx.x+threadIdx.x]);
c+=1.0f;
}
if(threadIdx.x > 0)
{
t += (data1[NX*blockIdx.x+threadIdx.x-1] - data1[NX*blockIdx.x+threadIdx.x]);
c+=1.0f;
}
if(threadIdx.x < NX-1)
{
t += (data1[NX*blockIdx.x+threadIdx.x+1] - data1[NX*blockIdx.x+threadIdx.x]);
c+=1.0f;
}
//printf("block %i, %i, %i\n", blockIdx.x, threadIdx.x, 1024*blockIdx.x+threadIdx.x);
//data2[1024*blockIdx.x+threadIdx.x] = 2*data1[1024*blockIdx.x+threadIdx.x];
if(blockIdx.x == 0)
data2[NX*blockIdx.x+threadIdx.x] = 1.0;
else
data2[NX*blockIdx.x+threadIdx.x] = data1[NX*blockIdx.x+threadIdx.x] + t/c*DIFF_RATE;
return;
}