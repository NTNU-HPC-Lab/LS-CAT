#include "includes.h"

using namespace std;

#define nsamples 250000
#define threadsPerBlock 500
#define num_blocks 500

// function to count samples in circle using cpu
__global__ void count_samples_GPU(float *d_X, float *d_Y, int *d_countInBlocks, int num_block, int samples)
{
__shared__ int shared_blocks[500];            // shared memory for threads in the same block

int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * num_block;

int inCircle = 0;
for (int i = index; i < samples; i += stride) {
float xValue = d_X[i];
float yValue = d_Y[i];

if (xValue*xValue + yValue * yValue <= 1.0f) {
inCircle++;
}
}

shared_blocks[threadIdx.x] = inCircle;
__syncthreads();                               //  prevent RAW/WAR/WAW hazards

// Pick thread 0 for each block to collect all points from each Thread.
if (threadIdx.x == 0)
{
int totalInCircleForABlock = 0;
for (int j = 0; j < blockDim.x; j++)
{
totalInCircleForABlock += shared_blocks[j];
}
d_countInBlocks[blockIdx.x] = totalInCircleForABlock;
}
}