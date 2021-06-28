#include "includes.h"
__global__ void calculate_entropy(float *norm,float *entropy,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
//printf("%d\n",idx);
int tid=threadIdx.x;
if(idx<size && norm[idx] !=0){
entropy[idx]=-(norm[idx]*log10f(norm[idx]));
//printf("%d f3 %f \n",idx,entropy[idx]);
__syncthreads();
}
for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
entropy[idx] += entropy[idx+ stride];
}
// synchronize within threadblock
__syncthreads();
}

if (idx == 0){

printf("entropy %f\n",entropy[0]);
}
}