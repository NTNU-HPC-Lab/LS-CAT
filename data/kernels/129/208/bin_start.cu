#include "includes.h"
__global__ void bin_start(int *binStart, int *binEnd, int *partBin, int nparts)
{
// This kernel function was adapted from NVIDIA CUDA 5.5 Examples
// This software contains source code provided by NVIDIA Corporation
extern __shared__ int sharedBin[];    //blockSize + 1
int index = threadIdx.x + blockIdx.x*blockDim.x;
int bin;

// for a given bin index, the previous bins's index is stored in sharedBin
if (index < nparts) {
bin = partBin[index];

// Load bin data into shared memory so that we can look
// at neighboring particle's hash value without loading
// two bin values per thread
sharedBin[threadIdx.x + 1] = bin;

if (index > 0 && threadIdx.x == 0) {
// first thread in block must load neighbor particle bin
sharedBin[0] = partBin[index - 1];
}
}
__syncthreads();

if (index < nparts) {
// If this particle has a different cell index to the previous
// particle then it must be the first particle in the cell,
// so store the index of this particle in the cell.
// As it isn't the first particle, it must also be the cell end of
// the previous particle's cell
bin = partBin[index];

if (index == 0 || bin != sharedBin[threadIdx.x]) {
binStart[bin] = index;

if (index > 0)
binEnd[sharedBin[threadIdx.x]] = index;
}

if (index == nparts - 1)
{
binEnd[bin] = index + 1;
}
}
}