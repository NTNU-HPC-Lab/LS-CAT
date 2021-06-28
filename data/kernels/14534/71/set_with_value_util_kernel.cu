#include "includes.h"
__global__ void set_with_value_util_kernel( double2 * __restrict buf, double v, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
double2 val;
val.x = v;
val.y = v;
buf[elem_id] = val;
}
}