#include "includes.h"
__global__ void dropout_op(size_t sz, float_t* random_nums, float_t* data, float_t drop_rate, float_t scale)
{
size_t index = blockIdx.x*blockDim.x + threadIdx.x;
if(index < sz)
{
if(random_nums[index] <= drop_rate)
{
data[index] = 0;
}
else
{
data[index] *= scale;
}
}
}