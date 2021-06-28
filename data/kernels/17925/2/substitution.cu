#include "includes.h"

// Helper function for using CUDA to call kernel functions
cudaError_t cuda_code(float* , float*, int , int );
__device__ float sum = 0;

__global__ void substitution(int i, int N, float *row, float *matrix, float*resultVector) {
int j = i + blockIdx.x * blockDim.x + threadIdx.x;
//From previous line, "i" assigns the initial thread index, so threads are not
//created for indexes that will not affect the results
int ij;		//element i,j of the matrix
if (j > i && j < N)
{
ij = j + (N + 1)*i;
row[j] = matrix[ij] * resultVector[j];
atomicAdd(&sum, row[j]);
}
__syncthreads();//Barrier to wait all threads to finish their tasks
}