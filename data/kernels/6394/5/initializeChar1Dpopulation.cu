#include "includes.h"
__device__ float generateRandom( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__device__ float generateRandomc( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__global__ void initializeChar1Dpopulation(char *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division){
int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
if(populationIndex<(sizeofPopulation*sizeofChormosome)){
population[populationIndex]= (char) ((int) (generateRandomc(globalState)*2)+48);
//printf("CUDA %d\n",population[populationIndex]);

}
__syncthreads();
}