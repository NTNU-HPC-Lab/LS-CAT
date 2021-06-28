#include "includes.h"
__global__ void naiveHistKernel(int* bins, int nbins, int* in, int nrows) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
auto offset = blockIdx.y * nrows;
auto binOffset = blockIdx.y * nbins;
for (; tid < nrows; tid += stride) {
int id = in[offset + tid];
if (id < 0)
id = 0;
else if (id >= nbins)
id = nbins - 1;
in[offset + tid] = id;
atomicAdd(bins + binOffset + id, 1);
}
}