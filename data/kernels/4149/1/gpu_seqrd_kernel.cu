#include "includes.h"
__global__ void gpu_seqrd_kernel(int *buffer, size_t reps, size_t elements)
{
int errors = 0;
for(size_t j = 0; j < reps; j++) {
size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
size_t step = blockDim.x * gridDim.x;
while(ofs < elements) {
// manually unroll loop to get multiple loads in flight per thread
int val1 = buffer[ofs];
ofs += step;
int val2 = (ofs < elements) ? buffer[ofs] : 0;
ofs += step;
int val3 = (ofs < elements) ? buffer[ofs] : 0;
ofs += step;
int val4 = (ofs < elements) ? buffer[ofs] : 0;
ofs += step;
// now check result of all the reads
if(val1 != 0) errors++;
if(val2 != 0) errors++;
if(val3 != 0) errors++;
if(val4 != 0) errors++;
}
}
if(errors > 0)
buffer[0] = errors;
}