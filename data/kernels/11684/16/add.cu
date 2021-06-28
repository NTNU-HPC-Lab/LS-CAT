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
__global__ void add(int n, float *a, float *b, float *result)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
result[i] = a[i] + b[i];
}

}