#include "includes.h"


// Device input vectors
int *d_a;
//Device output vector
int *d_b;








__global__ void downSweep(int *A, int size, int iteration) {
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < size) {
int aux;
if (!((index + 1) % (1 << (iteration + 1)))) {
aux = A[index - (1<<iteration)];
A[index - (1<<iteration)] = A[index];
A[index] = aux + A[index];
}
}
}