#include "includes.h"
__global__ void check_done_kernel(bool *mask, int num_vtx, bool *finished) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
while (*finished && tid < num_vtx) {
if (mask[tid])
*finished = false;
tid += blockDim.x * gridDim.x;
}

}