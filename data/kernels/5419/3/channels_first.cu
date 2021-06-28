#include "includes.h"
__global__ void channels_first(float* input, float* rinput, int channels, int height, int width, int pad_size)
{
// n (batch size), c (num of channels), y (height), x (width)
int n = blockIdx.x;
int y = blockIdx.y;
int x = blockIdx.z;

int ch_off = threadIdx.x;
float value;

int dimcyx = channels * height * width;
int dimyx = height * width;

int p_dimx = (width + 2 * pad_size);
int p_dimy = (height + 2 * pad_size);
int p_dimyxc = channels * p_dimy * p_dimx;
int p_dimxc = p_dimx * channels;

for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
value = input[n * dimcyx + c * dimyx + y * width + x];
rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
}
}