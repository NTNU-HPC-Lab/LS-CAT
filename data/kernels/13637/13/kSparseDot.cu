#include "includes.h"
__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha) {

const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row < m && col < n) {
const int start = indptr[row];
const int end = indptr[row + 1];
float sum = 0;
for (int i = start; i < end; i++) {
sum += data[i]  * dense_data[col * k + indices[i]];
}
const int pos = col * m + row;
target[pos] = alpha * sum + ((beta == 0) ? 0 : beta * target[pos]);
}
}