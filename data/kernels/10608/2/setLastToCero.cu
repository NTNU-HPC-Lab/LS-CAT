#include "includes.h"


// Device input vectors
int *d_a;
//Device output vector
int *d_b;








__global__ void setLastToCero(int *A, int size) {
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index == size - 1) {
A[index] = 0;
}
}