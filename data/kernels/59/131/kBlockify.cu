#include "includes.h"
__global__ void kBlockify(float* source, float* target, int numdims, int blocksize) {
const unsigned int idx = threadIdx.x;
const unsigned int numThreads = blockDim.x;
const int off = blockIdx.x * numdims;

for (unsigned int target_ind = idx; target_ind < numdims; target_ind += numThreads) {
const int block = target_ind / blocksize;
target[off + target_ind] = source[off + block * blocksize];
}
}