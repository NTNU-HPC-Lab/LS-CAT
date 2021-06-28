#include "includes.h"
__global__ void gpu_rndrd_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
// we don't want completely random writes here, since the performance would be awful
// instead, let each warp move around randomly, but keep the warp coalesced on 128B-aligned
//  accesses
int errors = 0;
for(size_t j = 0; j < reps; j++) {
// starting point is naturally aligned
size_t p = blockIdx.x * blockDim.x + threadIdx.x;
// if we start outside the block, sit this out (just to keep small runs from crashing)
if(p >= elements) break;

// quadratic stepping via "acceleration" and "velocity"
size_t a = 548191;
size_t v = 24819 + (p >> 5);  // velocity has to be different for each warp

for(size_t i = 0; i < steps; i += 4) {
// delta is multiplied by 32 elements so warp stays converged (velocity is the
//  same for all threads in the warp)
// manually unroll loop to get multiple loads in flight per thread
size_t p0 = p;
p = (p + (v << 5)) % elements;
v = (v + a) % elements;
size_t p1 = p;
p = (p + (v << 5)) % elements;
v = (v + a) % elements;
size_t p2 = p;
p = (p + (v << 5)) % elements;
v = (v + a) % elements;
size_t p3 = p;
p = (p + (v << 5)) % elements;
v = (v + a) % elements;

int v0 = buffer[p0];
int v1 = buffer[p1];
int v2 = buffer[p2];
int v3 = buffer[p3];

if(v0 != p1) errors++;
if(v1 != p2) errors++;
if(v2 != p3) errors++;
if(v3 != p) errors++;
}
}
if((errors > 0) && (reps > elements))
buffer[0] = errors;
}