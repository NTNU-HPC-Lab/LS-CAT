#include "includes.h"
__global__ void scale_channels_kernel(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size) {
if (scale_wh) {
int osd_index = index % channel_size + (index / batch_size)*channel_size;

out[index] = in_w_h_c[index] * scales_c[osd_index];
}
else {
out[index] = in_w_h_c[index] * scales_c[index / channel_size];
}
}
}