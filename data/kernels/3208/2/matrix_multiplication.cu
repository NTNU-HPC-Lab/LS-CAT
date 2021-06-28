#include "includes.h"
__global__ void matrix_multiplication(int *matrix_1, int *matrix_2, int *matrix_r, int m, int n, int p){

int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Multiply this row...
int col = threadIdx.x + blockIdx.x * blockDim.x;	// with this column.

// Matrix multiplication as follows:
// (m x n) x (n x p) = (m x p)

int id = row * p + col;	// Index of the result matrix in which we will write.
int sum = 0;

if (row < m && col < p) {
for(int i = 0; i < n; i++) {
// In matrix_1 we keep the row and advance in the columns.
// In matrix_2 we keep the column and advance in the rows.
sum = sum + matrix_1[row * n + i] * matrix_2[i * p + col];
// row * n stays in the same row and  "+ i" advances 1 column each cicle.
// i * p advances one row each cicle and  "+ col" keeps the same col.
}
matrix_r[id] = sum;
}
}