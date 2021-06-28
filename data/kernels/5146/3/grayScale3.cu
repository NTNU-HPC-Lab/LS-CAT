#include "includes.h"
__global__ void grayScale3(uchar3 *input, uchar3 *output,int width, int height) {


int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

//if ((gridDim.x * gridDim.y) < width * height){

int tid = y*width + x;

if (x<width){

if (y<height){


output[tid].x = (input[tid].x + input[tid].y +
input[tid].z) / 3;

output[tid].z = output[tid].y = output[tid].x;

}
}

// }
}