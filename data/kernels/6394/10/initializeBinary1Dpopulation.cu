#include "includes.h"
__device__ float generateRandom( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__global__ void initializeBinary1Dpopulation(int *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division){
int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
if(populationIndex<(sizeofPopulation*sizeofChormosome)){
population[populationIndex]=(int) (generateRandom(globalState)*2);
//printf("CUDA %d\n",population[populationIndex]);

}
__syncthreads();
}