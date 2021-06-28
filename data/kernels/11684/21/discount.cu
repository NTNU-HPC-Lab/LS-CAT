#include "includes.h"
extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"


extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

/*
* Perfom a reduction from data of length 'size' to result, where length of result will be 'number of blocks'.
*/
extern "C"
__global__ void discount(int n, float *a, float *b, float p, float *result)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
result[i] = a[i] / (1.0f + b[i] * p);
}
}