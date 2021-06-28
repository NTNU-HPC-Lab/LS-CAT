#include "includes.h"
__global__ void matrix_multiply_cuda(int* d_a, int* d_b, int* d_c, int m, int n) {

int i = blockIdx.y * blockDim.y + threadIdx.y;    // Row i of matrix C
int j = blockIdx.x * blockDim.x + threadIdx.x;    // Column j of matrix C

//Compute c[i][j] = a[i][k]+b[k][j] over k = 0...n-1
int cell = 0;
for (int k=0; k<n; k++)
cell += d_a[i*n+k] * d_b[k*m+j];
d_c[i*m+j]=cell;
}