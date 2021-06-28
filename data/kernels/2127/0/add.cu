#include "includes.h"

__global__ void add(int n, float *x, float *y) {

// Calculate the starting value for the for loop's index
int index = ( blockDim.x * blockIdx.x ) + threadIdx.x;

// Calculate the stride between elements of the arrays
int stride = blockDim.x * gridDim.x;

// Add the elements from array x and array y within the block
for (int i = index; i < n; i += stride)
// Store the result in y
y[i] = x[i] + y[i];
}