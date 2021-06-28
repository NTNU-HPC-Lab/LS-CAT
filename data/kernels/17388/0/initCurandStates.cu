#include "includes.h"

/*
* File smc_kernels.cu contains definitions of kernels used by SMC.
*/

#ifdef __NVCC__




#endif
__global__ void initCurandStates(curandState* randStates, int numThreads, int seed) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= numThreads || i < 0) return;

// Double check this seed, need only to be unique over one inference, as time should vary between inferences.
// curand_init(1234 + clock64(), seed * numThreads + i, 0, &particles->randStates[i]);
curandState randStateLocal = randStates[i];
curand_init(1234 + clock64(), seed * numThreads + i, 0, &randStateLocal);
randStates[i] = randStateLocal;
}