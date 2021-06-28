#include "includes.h"
__global__ void gpu_seqwr_kernel(int *buffer, size_t reps, size_t elements)
{
for(size_t j = 0; j < reps; j++) {
size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
size_t step = blockDim.x * gridDim.x;
while(ofs < elements) {
buffer[ofs] = 0;
ofs += step;
}
}
}