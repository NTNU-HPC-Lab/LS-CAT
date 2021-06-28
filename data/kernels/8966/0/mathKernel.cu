#include "includes.h"

// Lets you use the Cuda FFT library



cudaError_t mathWithCuda(float *output, float *input1, float *input2, unsigned int size, int oper);

// Using __global__ to declare function as device code (GPU)
// Do the math inside here:

// Helper function for using CUDA to add vectors in parallel.
__global__ void mathKernel(float *output, float *input1, float *input2, int n, int oper)
{
// Allocate elements to threads
int i = threadIdx.x + blockIdx.x * blockDim.x;

// Avoid access beyond the end of the array
if (i < n)
{
// No for-loop needed, CUDA runtime will thread this
switch (oper)
{
case 1: // Addition
output[i] = input1[i] + input2[i];
break;
case 2: // Subtraction
output[i] = input1[i] - input2[i];
break;
case 3: // Multiplication
output[i] = input1[i] * input2[i];
break;
case 4: // Division
output[i] = input1[i] / input2[i];
break;

// Add more operations here:
case 5:
break;
case 6:
break;
case 7:
break;

default:
return;
}

// Ensure all the data is available
__syncthreads(); // Gives a syntax "error" but this doesn't give build errors
}
}