#include "includes.h"
__global__ void calculate_contrast(float *norm,float *contrast,int *dif,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
int tid=threadIdx.x;
//printf("%d\n",tid);
if (idx >= max*max) return;
// in-place reduction in global memory
//float *contrast=norm+blockIdx.x*blockDim.x;
if(idx<size){
contrast[idx]=norm[idx]*dif[idx];
//printf("%f %f\n",norm[idx],contrast[idx]);
__syncthreads();
}
for (int stride = 1; stride < max; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
contrast[idx] += contrast[idx+ stride];
//printf("%d %f\n",idx,contrast[idx]);
}
// synchronize within threadblock
__syncthreads();
}

if (idx == 0){
printf("contrast %f\n",contrast[0]);
}
}