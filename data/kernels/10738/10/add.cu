#include "includes.h"
__global__ void add(int *a, int *b, int *c,int columns,int rows)
{
// get the global id for the thread
int x = (blockIdx.x * blockDim.x + threadIdx.x);
int y = (blockIdx.y * blockDim.y + threadIdx.y);

// calculate the index of the input data
int index = y * columns + x;

c[index] = a[index] + b[index];
}