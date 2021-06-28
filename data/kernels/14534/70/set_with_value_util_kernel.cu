#include "includes.h"
__global__ void set_with_value_util_kernel( int4 * __restrict buf, int v, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
int4 val;
val.x = v;
val.y = v;
val.z = v;
val.w = v;
buf[elem_id] = val;
}
}