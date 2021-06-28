#include "includes.h"
__global__ void ElementWiseMultiply_CUDA(double *C, double *A, double *B, int rows, int cols)
{
int j = blockDim.x * blockIdx.x + threadIdx.x;
int i = blockDim.y * blockIdx.y + threadIdx.y;

int sourceLength = cols * rows;
int sourceIndex = i + (j * blockDim.y);
int targetIndex = i + (j * blockDim.y);
if ((sourceIndex <= sourceLength - 1) & (targetIndex < rows))
{
//if (i == 0 & j == 0)
//{
//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
//}
//int idx = i + (j * dimC.y);
double a = A[sourceIndex];
double b = B[sourceIndex];
C[targetIndex] = a * b;
//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
}
}