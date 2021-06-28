#include "includes.h"

using namespace std;




__global__ void propagateCarries(int* d_matrix, int numCols) {
int idx = blockDim.x * blockIdx.x + threadIdx.x * numCols;
int carry = 0;

for (int i = numCols - 1; i >= 0; i--) {
int rowVal = (d_matrix[idx + i] + carry) % 10;
carry = (d_matrix[idx + i] + carry) / 10;

d_matrix[idx + i] = rowVal;
}
}