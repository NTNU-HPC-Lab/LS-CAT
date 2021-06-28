#include "includes.h"
__global__ void THCudaTensor_kernel_copy(float *dst, long *dst_sz, long *dst_st, int dst_dim, float *src, long *src_sz, long *src_st, int src_dim, long n_elem, long innerdim)
{
long k = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x)*blockDim.y + threadIdx.y;

long i_start = threadIdx.x * src_st[src_dim-1];
long i_step = blockDim.x * src_st[src_dim-1];

long o_start = threadIdx.x * dst_st[dst_dim-1];
long o_step = blockDim.x * dst_st[dst_dim-1];
long o_end = innerdim * dst_st[dst_dim-1];

if ( ((k+1) * innerdim) <= n_elem) // too safe
{
long dst_idx = 0;
long dst_rest = k * innerdim;
for(int dim = 0; dim < dst_dim; dim++)
{
dst_idx += (dst_rest/dst_sz[dim])*dst_st[dim];
dst_rest = dst_rest % dst_sz[dim];
}

long src_idx = 0;
long src_rest = k * innerdim;
for(int dim = 0; dim < src_dim; dim++)
{
src_idx += (src_rest/src_sz[dim])*src_st[dim];
src_rest = src_rest % src_sz[dim];
}

for (int i=i_start, o=o_start; o<o_end; i+=i_step, o+=o_step) {
dst[dst_idx + o] = src[src_idx + i];
}
}
}