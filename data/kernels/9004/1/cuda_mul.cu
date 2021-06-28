#include "includes.h"

#define MAT_TYPE double
#define MAT_SIZE 1024
#define N MAT_SIZE
#define N2 MAT_SIZE*MAT_SIZE

#define BLOCK 256
#define THREAD 512

void stopwatch(int);






__global__ void cuda_mul(MAT_TYPE* A,MAT_TYPE* B,MAT_TYPE* C,int w)
{
int tid,tx,ty;

tx = blockDim.x * blockIdx.x + threadIdx.x;
ty = blockDim.y * blockIdx.y + threadIdx.y;
tid = w*ty + tx;

MAT_TYPE v = 0;
MAT_TYPE a = 0;
MAT_TYPE b = 0;

for(int i=0;i< w;i++)
{
a = A[ty * w + i];
b = B[i * w + tx];
v += a+b;
}

C[tid]= v;
}