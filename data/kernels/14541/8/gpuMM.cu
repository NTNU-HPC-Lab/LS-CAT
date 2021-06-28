#include "includes.h"
__global__ void gpuMM(float *A, float *B, float *C, int N)
{
// Matrix multiplication for NxN matrices C=A*B
// Each thread computes a single element of C
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

float sum = 0.0;
for (int n = 0; n < N; ++n)
sum += A[row*N+n]*B[n*N+col];

C[row*N+col] = sum;

//	if(row%50 ==5)
//		printf("%f \t %f \t %f\n",A[row*N+col], B[row*N+col], C[row*N+col]);

}