#include "includes.h"
__global__ void filterKernel(unsigned char* data, unsigned width, unsigned height, unsigned hStride, unsigned vStride, bool wrapAround) {
unsigned columnId = blockIdx.x * blockDim.x + threadIdx.x;

if (columnId < width) {
unsigned char* colp = data + columnId * hStride;
unsigned step = width * hStride * vStride;
{
uint32_t prev = colp[step];
// boundary condition
{
uint32_t pprev;
if (wrapAround) {
pprev = colp[step * (height - 1)];
} else {
pprev = prev;
}
uint32_t v = colp[0];
colp[0] = (2 * v + pprev + prev) >> 2;
}
__syncthreads();  // because of if
for (unsigned row = 2; row < height; row += 2) {
uint32_t next = colp[step * (row + 1)];
uint32_t v = colp[step * row];
colp[step * row] = (2 * v + next + prev) >> 2;
prev = next;
}
}
}
}