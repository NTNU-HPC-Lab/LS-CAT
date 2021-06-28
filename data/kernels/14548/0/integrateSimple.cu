#include "includes.h"


#define INTERVALS 1000000

// Max number of threads per block
#define THREADS 512
#define BLOCKS 64

double calculatePiCPU();

// Synchronous error checking call. Enable with nvcc -DDEBUG
__global__ void integrateSimple(float *sum, float step, int threads, int blocks)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

for (int i = idx; i < INTERVALS; i+=threads*blocks)
{
float x = (i+0.5f) * step;
sum[idx] += 4.0f / (1.0f+ x*x);
}
}