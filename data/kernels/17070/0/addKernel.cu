#include "includes.h"



// Helper function for using CUDA to add vectors in parallel.
__global__ void addKernel(int* c, const int* a, const int* b, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) {
c[i] = a[i] + b[i];
}
}