#include "includes.h"


#define _SIZE_ 1000000

/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

*/


__global__ void addLoopGPU(int* a, int* b, int* c)
{
int tid = blockIdx.x;
if (tid < 64)
c[tid] = abs(powf(b[tid], 2) - powf(b[tid], 2));
}