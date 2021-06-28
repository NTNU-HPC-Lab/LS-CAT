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

__global__ void total(float *input, float *output, int len)
{
//@@ Load a segment of the input vector into shared memory
//@@ Traverse the reduction tree
//@@ Write the computed sum of the block to the output vector at the
//@@ correct index
__shared__ float partSum[2 * BLOCK_SIZE];
unsigned int tx = threadIdx.x;
unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
//Loading input floats to shared memory
//Take care of the boundary conditions
if (start + tx < len){
partSum[tx] = input[start + tx];
if (start + BLOCK_SIZE + tx <len) partSum[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
else partSum[BLOCK_SIZE + tx] = 0;
}
else{
partSum[tx] = 0;
partSum[BLOCK_SIZE + tx] = 0;
}
unsigned int stride;
for (stride = BLOCK_SIZE; stride > 0; stride = stride / 2){
__syncthreads();
if (tx < stride) partSum[tx] += partSum[tx + stride];
}
if (tx == 0){ output[blockIdx.x] = partSum[0]; }
}