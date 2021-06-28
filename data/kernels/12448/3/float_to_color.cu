#include "includes.h"
__global__ void float_to_color(uchar4 * pixels, float* in){
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

float num = in[offset];

pixels[offset].x = (int)(num*255);
pixels[offset].y = (int)(0);
pixels[offset].z = (int)((MAX_TEMP-num) * 255);
pixels[offset].w = 255;
}