#include "includes.h"
__global__ void calculate_idm(float *norm,float *idm,int*dif,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
//printf("%d\n",idx);
int tid=threadIdx.x;
if(idx<size){
idm[idx]=((float(1)/(1+dif[idx]))*(norm[idx]));
//printf("%d  %f %f %f\n",idx,idm[idx],norm[idx],(float(1)/(1+dif[idx])));
__syncthreads();
}
for (int stride = 1; stride < blockDim.x; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
idm[idx] += idm[idx+ stride];
//printf("%d %f\n",idx,idm[idx]);
}
// synchronize within threadblock
__syncthreads();
}

if (idx == 0){

printf("idm %f\n",idm[0]);
}
}