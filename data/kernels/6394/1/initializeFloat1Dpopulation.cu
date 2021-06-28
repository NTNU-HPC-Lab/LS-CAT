#include "includes.h"



using namespace std;


float *valuesf;
float *weightf;
float maxWf;

float *matchf;
const int fSUMFLAG=0;
const int fKNAPSACKFLAG = 1;

const int fAVGFLAG=2;
const int fMATCHFLAG=3;
const int fINVERSESUMFLAG=4;

const int fMAXIMIZE=-1;
const int fMINIMIZE=1;




__device__ float generateRandom( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__device__ float generateRandomf( curandState* globalState)
{
//int ind = threadIdx.x;
curandState localState = globalState[0];
float RANDOM = curand_uniform( &localState );
globalState[0] = localState;
return RANDOM;
}
__global__ void initializeFloat1Dpopulation(float *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division){
int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
if(populationIndex<(sizeofPopulation*sizeofChormosome)){
population[populationIndex]= (int)(generateRandomf(globalState)*2);
//printf("CUDA %d\n",population[populationIndex]);

}
__syncthreads();
}