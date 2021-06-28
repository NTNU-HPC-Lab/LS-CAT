#include "includes.h"
__global__ void kernel_vec_equals_minus_vec1(double      *vec, double      *vec1, int numElements)
{

int iam = threadIdx.x;
int bid = blockIdx.x;
int threads_in_block = blockDim.x;
int gid = bid*threads_in_block + iam;

if (gid < numElements){
vec[gid] = -vec1[gid];
}
}