#include "includes.h"
__global__ void kernel_s(unsigned int * ind, const size_t nbn, const unsigned int ne)
{
int m_i_b = threadIdx.x;
if (m_i_b >= ne)  return;
extern __shared__ float dats[];
dats[m_i_b] = ind[m_i_b];
__syncthreads();

for (int q = 1; q < nbn; q *= 2) {
if (m_i_b >= q) {
dats[m_i_b] += dats[m_i_b - q];
}
__syncthreads();
}
if (m_i_b == 0)  ind[0] = 0;
else  ind[m_i_b] = dats[m_i_b - 1];
}