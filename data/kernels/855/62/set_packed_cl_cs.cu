#include "includes.h"
__global__ void set_packed_cl_cs(int *d_packed_cl, int *d_packed_cs, int *d_cl, int *d_cs, int *d_gcs, int chunk_num)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= chunk_num) {
return;
}

if (d_gcs[i + 1] - d_gcs[i] > 0) {
d_packed_cl[d_gcs[i]] = d_cl[i];
d_packed_cs[d_gcs[i]] = d_cs[i];
}
}