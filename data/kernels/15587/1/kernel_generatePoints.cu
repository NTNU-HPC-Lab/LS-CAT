#include "includes.h"
__global__ void kernel_generatePoints( curandState * globalState, int* counts, int totalNumThreads)
{
int index = (blockIdx.x * blockDim.x) + threadIdx.x;
float x,y;
if(index >= totalNumThreads){
return;
}
curandState localState = globalState[index];
for(int i = 0; i < NUM_POINTS_PER_THREAD; i++)
{
x = curand_uniform( &localState);
y = curand_uniform( &localState);
if(x*x+y*y <=1){
counts[index]++;
}
}
globalState[index] = localState;
}