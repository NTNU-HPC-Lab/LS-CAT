#include "includes.h"
__global__ void affine_transform(size_t sz, float_t* audio, float_t* end_out, size_t stride)
{
size_t index = blockDim.x * blockIdx.x + threadIdx.x;

if(index < sz)
{
audio[index+stride] = (audio[index+stride]-end_out[index])/expf(end_out[index+stride]);
}
}