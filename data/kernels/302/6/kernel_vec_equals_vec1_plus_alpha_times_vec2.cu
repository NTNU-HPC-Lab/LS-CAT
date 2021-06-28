#include "includes.h"
__global__ void kernel_vec_equals_vec1_plus_alpha_times_vec2(double      *vec, double      *vec1, double       alpha, double      *d_a1, double      *vec2, int numElements)
{
int iam = threadIdx.x;
int bid = blockIdx.x;
int threads_in_block = blockDim.x;
int gid = bid*threads_in_block + iam;

if (gid < numElements){
double a = alpha;
if (d_a1) a *= *d_a1;

vec[gid] = vec1[gid] + a * vec2[gid];
}
}