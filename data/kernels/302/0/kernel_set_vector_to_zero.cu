#include "includes.h"
__global__ void kernel_set_vector_to_zero(double *d_vec, int dimension)
{

int iam = threadIdx.x;
int bid = blockIdx.x;
int threads_in_block = blockDim.x;
int gid = bid*threads_in_block + iam;

if (gid < dimension){
d_vec[gid] = 0;
}
}