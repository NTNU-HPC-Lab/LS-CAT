#include "includes.h"
__global__ void calculate_ASM(float *norm,float *ASM,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
int tid=threadIdx.x;
if(idx<size){
ASM[idx]=norm[idx]*norm[idx];
// printf("%d asm %f\n",idx,norm[idx]);
}
//corelation[idx]=(((i*j)*norm[idx]));

for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{

ASM[idx] += ASM[idx+stride];
//printf("%d %f %f\n",idx,corelation[idx],ASM[idx]);
}
// synchronize within threadblock
__syncthreads();
}

if (idx == 0){

printf("ASM %f %d\n",ASM[0],idx);
}
}