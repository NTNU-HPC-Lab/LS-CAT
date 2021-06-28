#include "includes.h"
__global__ void kernel_m(unsigned int * ind, unsigned int *scand, unsigned int shift, const unsigned int ne)
{
unsigned int sosm = 1 << shift;
int m_i_b = threadIdx.x + blockDim.x * blockIdx.x;
if (m_i_b >= ne)  return;
scand[m_i_b] = ((ind[m_i_b] & sosm) >> shift) ? 0 : 1;
}