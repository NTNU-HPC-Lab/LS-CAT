#include "includes.h"
__global__ void getRow_naive(const float * A, float * row_id, float * out, int Acols) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < Acols) {
out[id] = A[id + (int)(*row_id)*Acols];
}
}