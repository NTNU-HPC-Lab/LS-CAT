#include "includes.h"
__global__ void kernel_initializeRand( curandState * randomGeneratorStateArray, unsigned long seed, int totalNumThreads)
{
int id = (blockIdx.x * blockDim.x) + threadIdx.x;
if( id >= totalNumThreads){
return;
}
curand_init( seed, id, 0, &randomGeneratorStateArray[id]);
}