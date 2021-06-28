#include "includes.h"
__device__ float rowcol_dot(float * matrix_a, float * matrix_b, int row, int col, int N)
{
float val = 0;

for (int k = 0; k < N; k++)
{
val += matrix_a[row*N + k] * matrix_b[col + k*N];
}
return val;
}
__global__ void matrix_mult_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

output_matrix[col + row * N] = rowcol_dot(matrix_a, matrix_b, row, col, N);
}