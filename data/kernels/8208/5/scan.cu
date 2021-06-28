#include "includes.h"
__global__ void scan(float *input, float *output, float *aux, int len) {

//@@declaring shared memeory of size 2*inputSize
__shared__ float XY[2 * BLOCK_SIZE];

//@@X-axis block id
int bx = blockIdx.x;

//@@X-axis thread id
int tx = threadIdx.x;

int i = 2 * bx * blockDim.x + tx;

//@@ loading data from global memory to shared memory stage 1
if (i<len)
XY[tx] = input[i];

//@@ loading data from global memory to shared memory stage 2
if (i + blockDim.x<len)
XY[tx + blockDim.x] = input[i + blockDim.x];

//@@making sure that all threads in a block are done with loading data from global memory to shared memory
//@@before proceeding to the calculations phase
__syncthreads();

for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
//@@making sure that all threads in a block are done with previous step before starting the next
__syncthreads();

int index = (tx + 1)*stride * 2 - 1;

if (index < 2 * BLOCK_SIZE)
XY[index] += XY[index - stride];
}

for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
//@@making sure that all threads in a block are done with previous step before starting the next
__syncthreads();

int index = (tx + 1)*stride * 2 - 1;

if (index + stride < 2 * BLOCK_SIZE)
XY[index + stride] += XY[index];
}

//@@making sure that all threads in a block are done with previous step before starting the next
__syncthreads();

if (i < len)
output[i] = XY[tx];

if (i + blockDim.x < len)
output[i + blockDim.x] = XY[tx + blockDim.x];

//@@storing the block sum to the aux array
if (aux != NULL && tx == 0)
aux[bx] = XY[2 * blockDim.x - 1];
}