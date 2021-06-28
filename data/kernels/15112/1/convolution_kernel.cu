#include "includes.h"

#define Width 1920
#define Height 2520
#define iterations 100




__global__ void convolution_kernel(unsigned char* A, unsigned char* B)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int x = i-2*blockIdx.x-1;
int y = j-2*blockIdx.y-1;

__shared__ unsigned char As[32][32];

//Copy from global memory to shared memory

if (x<0) {
x=0;
} else if (x==Width) {
x=Width-1;
}
if (y<0) {
y=0;
} else if (y == Height) {
y = Height-1;
}
As[threadIdx.x][threadIdx.y] = A[Width*y + x];

__syncthreads();

// Computations

if (threadIdx.x!=0 && threadIdx.x!=31 && threadIdx.y!=0 && threadIdx.y!=31) {
B[Width*y + x] =     (As[threadIdx.x-1][threadIdx.y-1]  +
As[threadIdx.x  ][threadIdx.y-1] * 2 +
As[threadIdx.x+1][threadIdx.y-1]  +
As[threadIdx.x-1][threadIdx.y  ] *2 +
As[threadIdx.x  ][threadIdx.y  ] *4 +
As[threadIdx.x+1][threadIdx.y  ] * 2 +
As[threadIdx.x-1][threadIdx.y+1] * 1 +
As[threadIdx.x  ][threadIdx.y+1] * 2 +
As[threadIdx.x+1][threadIdx.y+1] * 1)/16;
}
}