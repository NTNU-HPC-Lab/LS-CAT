#include "includes.h"
__global__ void blendingGray(uchar3 *input, uchar3 *input2, uchar3 *output,int width, int height,float coefficient) {


int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;


int tid = y*width + x;

int nbPixels = width * height;
float prod = coefficient * (float) nbPixels;
int prodfin = (int) prod;


if (x<width){

if (y<height){

if (tid <= prodfin){

output[tid].x = input[tid].x;

output[tid].z = output[tid].y = output[tid].x;

}
else{

output[tid].x = input2[tid].x;

output[tid].z = output[tid].y = output[tid].x;

}

}
}


}