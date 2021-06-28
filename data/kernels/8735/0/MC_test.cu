#include "includes.h"
/**************************************
***************************************
* Code Can be compiled using --> nvcc kernel5.cu -lcurand if the cuRand lib is the envirement PATH
* else use nvcc kernel5.cu -L</path/to/the/lib> -lcurand
***************************************
**************************************/



__global__ void MC_test(unsigned int seed,curandState *states,unsigned int numsim,unsigned int *results)
{
extern __shared__ int sdata[];
int i;
int nthreads = gridDim.x * blockDim.x;
unsigned int innerpoint=0;
int tx=threadIdx.x;
int idx = blockIdx.x * blockDim.x + tx;
curandState *state =states + idx;
float x,y,l2norm2;
sdata[tx]=0;
__syncthreads();
curand_init(seed, tx, 0, state);
__syncthreads();
for(i=tx;i<numsim;i+=nthreads){
x = curand_uniform(state);
y = curand_uniform(state);
l2norm2 = x * x + y * y;
if (l2norm2 < static_cast<float>(1))
{
innerpoint++;;
}
}
__syncthreads();
sdata[tx]=innerpoint;
__syncthreads();
//-------reduction
for (unsigned int s=blockDim.x/2;s>0;s>>=1){
if(tx < s){
sdata[tx]=sdata[tx]+sdata[tx+s];
}
}
//-----------------
__syncthreads();
if(tx==0){
results[blockIdx.x]=sdata[0];
}

}