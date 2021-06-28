#include "includes.h"
__global__ void transposeSmemPad(float *out, float *in, int nx, int ny)
{
// static shared memory with padding
__shared__ float tile[BDIMY][BDIMX + IPAD];

// coordinate in original matrix
unsigned int  ix, iy, ti, to;
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;

// linear global memory index for original matrix
ti = iy * nx + ix;

// thread index in transposed block
unsigned int bidx, irow, icol;
bidx = threadIdx.y * blockDim.x + threadIdx.x;
irow = bidx / blockDim.y;
icol = bidx % blockDim.y;

// coordinate in transposed matrix
ix = blockDim.y * blockIdx.y + icol;
iy = blockDim.x * blockIdx.x + irow;

// linear global memory index for transposed matrix
to = iy * ny + ix;

// transpose with boundary test
if (ix < nx && iy < ny)
{
// load data from global memory to shared memory
tile[threadIdx.y][threadIdx.x] = in[ti];

// thread synchronization
__syncthreads();

// store data to global memory from shared memory
out[to] = tile[icol][irow];
}
}