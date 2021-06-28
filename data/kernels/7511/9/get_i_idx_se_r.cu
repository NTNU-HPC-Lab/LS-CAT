#include "includes.h"
__global__ void get_i_idx_se_r(const int nloc, const int * ilist, int * i_idx)
{
const unsigned int idy = blockIdx.x * blockDim.x + threadIdx.x;
if(idy >= nloc) {
return;
}
i_idx[ilist[idy]] = idy;
}