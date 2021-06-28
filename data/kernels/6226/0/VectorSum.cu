#include "includes.h"


cudaError_t sumWithCuda(int *c, const int *a, const int *b, unsigned int size);

//ÄÚºËº¯Êý

__global__ void VectorSum(int *result, const int *vector_a, const int *vector_b)
{
int i = threadIdx.x;
result[i] = vector_a[i] + vector_b[i];
printf("%d : call kernel function.\n", i);
}