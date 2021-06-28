#include "includes.h"
__global__ void grayScale2(uchar3 *input, uchar3 *output,int width, int height) {

int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int w = blockDim.x * gridDim.x;

//if ((gridDim.x * gridDim.y) < width * height){

int tid = y*w + x;

output[tid].x = (input[tid].x + input[tid].y +
input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;

// }
}