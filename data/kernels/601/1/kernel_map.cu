#include "includes.h"

#define NUM_THREADS 511
#define ITERATIONS 100000

using namespace std;



__global__ void kernel_map(int *values, int *next_values)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;

if (tid < NUM_THREADS)
{
next_values[tid] = values[tid] + 1;
}
}