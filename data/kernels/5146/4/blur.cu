#include "includes.h"
__global__ void blur(uchar3 *input, uchar3 *output,int width, int height) {

int matrix[7][7] = {{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,3,59,97,59,13,1},{2,22,97,159,97,22,2},{1,3,59,97,59,3,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};


int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

//if ((gridDim.x * gridDim.y) < width * height){

int tid = y*width + x;

int outputTemp = 0;

int sommeCoef = 0;

if (x<width){

if (y<height){

if (x>3 && x<width-3 && y>3 && y<height-3){

for (int i=0; i<7; i++){

for (int j=0; j<7; j++){

outputTemp += input[(y-3+i)*width+(x-3+j)].x*matrix[j][i];

sommeCoef += matrix[j][i];

}


output[tid].x = outputTemp / sommeCoef;

output[tid].z = output[tid].y = output[tid].x;

}
}

}

}

// }
}