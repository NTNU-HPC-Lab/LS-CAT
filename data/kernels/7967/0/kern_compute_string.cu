#include "includes.h"

// CUDA runtime

// includes

extern "C"
{
}
#define MEMSIZE 30


/* Function computing the final string to print */
__global__ void kern_compute_string(char *res, char *a, char *b, char *c, int length)
{
int i;
i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < length)
{
res[i] = a[i] + b[i] + c[i];
}
}