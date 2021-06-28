#include "includes.h"
// Include files


// Parameters

#define N_ATOMS 343
#define MASS_ATOM 1.0f
#define time_step 0.01f
#define L 10.5f
#define T 0.728f
#define NUM_STEPS 10000

const int BLOCK_SIZE = 1024;
//const int L = ;
const int scheme = 1; // 0 for explicit, 1 for implicit

/*************************************************************************************************************/
/*************								INITIALIZATION CODE										**********/
/*************************************************************************************************************/

__global__ void newForceReduction(float *input, float *output, int startunit, int len)
{
unsigned int tx = threadIdx.x;
unsigned int start = blockIdx.x *N_ATOMS;

__shared__ float partSum[BLOCK_SIZE];
// if (tx == 0) printf("Length of the shared memory array - %i \n", N_ATOMS);

//Loading input floats to shared memory
//Take care of the boundary conditions
if (tx < N_ATOMS) { partSum[tx] = input[start + tx]; }
else{ partSum[tx] = 0.0f; }

__syncthreads();

//Reduction Kernel for each dimension
if (tx < 512){
partSum[tx] += partSum[tx + 512];
} __syncthreads();
if (tx < 256){
partSum[tx] += partSum[tx + 256];
} __syncthreads();
if (tx < 128){
partSum[tx] += partSum[tx + 128];
} __syncthreads();
if (tx < 64){
partSum[tx] += partSum[tx + 64];
} __syncthreads();
if (tx < 32){
partSum[tx] += partSum[tx + 32];
partSum[tx] += partSum[tx + 16];
partSum[tx] += partSum[tx + 8];
partSum[tx] += partSum[tx + 4];
partSum[tx] += partSum[tx + 2];
partSum[tx] += partSum[tx + 1];
}
if (tx == 0){
output[blockIdx.x] = -partSum[0];
}
}