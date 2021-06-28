#include "includes.h"
__global__ void gpu_rndwr_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
// we don't want completely random writes here, since the performance would be awful
// instead, let each warp move around randomly, but keep the warp coalesced on 128B-aligned
//  accesses
for(size_t j = 0; j < reps; j++) {
// starting point is naturally aligned
size_t p = blockIdx.x * blockDim.x + threadIdx.x;
// if we start outside the block, sit this out (just to keep small runs from crashing)
if(p >= elements) break;

// quadratic stepping via "acceleration" and "velocity"
size_t a = 548191;
size_t v = 24819 + (p >> 5);  // velocity has to be different for each warp

for(size_t i = 0; i < steps; i++) {
size_t prev = p;
// delta is multiplied by 32 elements so warp stays converged (velocity is the
//  same for all threads in the warp)
p = (p + (v << 5)) % elements;
v = (v + a) % elements;
buffer[prev] = p;
}
}
}