#include "includes.h"




using namespace std;





#define N 32

__global__ void multSquareMatrix(int *A, int *B, int *result, int n)
{
int k, sum = 0;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

for (k = 0; k < n; k++) {
sum += A[row * n + k] * B[k * n + col];
result[row * n + col] = sum;
}
}