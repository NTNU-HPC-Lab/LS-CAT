#include "includes.h"
__global__ void find_min_kernel(float * d_out, const float * d_in)
{
// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
extern __shared__ float sdata[];

const int threadGId = blockIdx.x * blockDim.x + threadIdx.x;
const int threadLId = threadIdx.x;

// load shared mem from global mem
sdata[threadLId] = d_in[threadGId];
__syncthreads();            // make sure entire block is loaded!

// do reduction in shared mem
for (unsigned int blockHalfSize = blockDim.x / 2; blockHalfSize > 0; blockHalfSize >>= 1) {
if (threadLId < blockHalfSize) {
sdata[threadLId] = min(sdata[threadLId], sdata[threadLId + blockHalfSize]);
}
__syncthreads();        // make sure all adds at one stage are done!
}

// only thread 0 writes result for this block back to global mem
if (threadLId == 0)
{
d_out[blockIdx.x] = sdata[0];
}
}