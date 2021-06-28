#include "includes.h"
__global__ void repack_input_kernel_2(float *input, float *re_packed_input, int w, int h, int c)
{
//__shared__ uint32_t tmp[33 * 32];  // 33x32 is misaligned 32 x 32 to avoid bank conflicts

int index = blockIdx.x*blockDim.x + threadIdx.x;

const int items_per_channel = w * h;

int c_pack = index % 32;
int chan_index = index / 32;
int chan = (chan_index * 32) % c;
int i = (chan_index * 32) / c;

//for (chan = 0; chan < c; chan += 32)
{
//for (i = 0; i < items_per_channel; ++i)
if (i < items_per_channel)
{
//for (c_pack = 0; c_pack < 32; ++c_pack)
{
float src = input[(chan + c_pack)*items_per_channel + i];

re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
}
}
}
}