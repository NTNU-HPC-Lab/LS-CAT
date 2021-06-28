#include "includes.h"
__global__ void randomColouring (curandState* globalState, int *degreeCount, int n, int limit){

int i= blockDim.x * blockIdx.x + threadIdx.x;

curandState localState = globalState[i];
float RANDOM = curand_uniform( &localState );
globalState[i] = localState;

RANDOM *= (limit - 1 + 0.999999);
RANDOM += 1;

degreeCount[i] = (int) RANDOM;
}