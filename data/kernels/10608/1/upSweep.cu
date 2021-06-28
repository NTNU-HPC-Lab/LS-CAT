#include "includes.h"


// Device input vectors
int *d_a;
//Device output vector
int *d_b;








__global__ void upSweep(int *A, int size, int iteration) {
const int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < size) {
if (!((index + 1) % (1 << (iteration + 1))))
A[index] = A[index - (1<<iteration)] + A[index];
}

}