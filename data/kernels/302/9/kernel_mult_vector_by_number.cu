#include "includes.h"
__global__ void kernel_mult_vector_by_number(double      *vec, double       alpha, int numElements)
{
int iam = threadIdx.x;
int bid = blockIdx.x;
int threads_in_block = blockDim.x;
int gid = bid*threads_in_block + iam;

if (gid < numElements){
vec[gid] *= alpha;
}

}