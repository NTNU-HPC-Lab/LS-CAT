#include "includes.h"



/*
Location qualifiers

__global__

Defines a kernel.
Runs on the GPU, called from the CPU.
Executed with <<<dim3>>> arguments.


__device__
Runs on the GPU, called from the GPU.
Can be used for variables too.

__host__

Runs on the CPU, called from the CPU.

Qualifiers can be mixed
Eg __host__ __device__ foo()
Code compiled for both CPU and GPU
useful for testing

*/


__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}