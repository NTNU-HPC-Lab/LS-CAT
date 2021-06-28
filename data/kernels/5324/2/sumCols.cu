#include "includes.h"

using namespace std;




__global__ void sumCols(int* d_matrix, int* d_result, int numRows, int numCols) {
int sum = 0;

int idx = blockDim.x * blockIdx.x + threadIdx.x;

for (int i = 0; i < numRows; i++) {
sum += d_matrix[idx + (numCols * i)];
}

d_result[idx] = sum;
}