#include "includes.h"

//Cuda checks
__global__ void matrix_multiply_kernel(unsigned char *temp, unsigned char *matrix, float *kernal, int order, int middle, int windowSizeX, int windowSizeY){
//Find place in the execution
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
float sum = 0.0;
//If out of bounds, do nothing
if(y >= windowSizeY || x >= windowSizeX){
return;
}
//Else do function
for(int y2 = 0; y2 < order; y2++){
for(int x2 = 0; x2 < order; x2++){
int tempX = x - middle + x2, tempY = y - middle + y2;
if(tempX < 0){
tempX = 0;
}else if(tempX >= windowSizeX){
tempX = windowSizeX - 1;
}
if(tempY < 0){
tempY = 0;
}else if(tempY >= windowSizeY){
tempY = windowSizeY - 1;
}
sum += temp[(windowSizeX * tempY) + tempX] * kernal[(order * x2) + y2];
}
}
//Clamp the sum value
if(sum < 0){
sum = 0;
}else if(sum > 255){
sum = 255;
}
//Add sum value to matrix
matrix[(windowSizeX * y) + x] = (unsigned char) sum;

}