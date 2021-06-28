#include "includes.h"
__global__ void getRow_IntId_naive(const float * A, int row_id, float * out, int Acols) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < Acols) {
out[id] = A[id + row_id*Acols];
}
}