#include "includes.h"
__global__ void gpu_latency_setup_kernel(int *buffer, size_t delta, size_t elements)
{
size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
size_t step = blockDim.x * gridDim.x;
while(ofs < elements) {
size_t tgt = ofs + delta;
while(tgt >= elements)
tgt -= elements;
buffer[ofs] = tgt;
ofs += step;
}
}