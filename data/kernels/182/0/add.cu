#include "includes.h"


#define N 128*256
#define THREADS_PER_BLOCK 256
#define N_BLOCKS N/THREADS_PER_BLOCK

// Kernel to add N integers using threads and blocks

// Main program
__global__ void add(int *a, int *b, int *c){
int index = blockIdx.x * blockDim.x + threadIdx.x;

c[index] = a[index] + b[index];
}