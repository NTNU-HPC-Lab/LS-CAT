#include "includes.h"
__global__ void multiplyVectors(float* A, float* B, float*C,int WIDTH,int HEIGHT) {
int x = threadIdx.x + blockIdx.x*blockDim.x;
int y = threadIdx.y + blockIdx.y*blockDim.y;

if (x<WIDTH && y<HEIGHT) {

double result = 0.0;

for (int i=0;i<WIDTH;i++)
result+=A[y*WIDTH+i]*B[i*WIDTH+x];

C[y*WIDTH+x] = result;
}
}