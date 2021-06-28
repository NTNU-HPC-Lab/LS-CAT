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
__global__ void initializeChar1DRangepopulation(char *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division,char* range){
int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
if(populationIndex<(sizeofPopulation*sizeofChormosome)){
population[populationIndex]= range[(int) (generateRandomc(globalState)*sizeofChormosome)];
//printf("CUDA %d\n",population[populationIndex]);

}
__syncthreads();
}