#include "includes.h"


#define index(i, j, w)  ((i)*(w)) + (j)


__global__ void blurKernel (unsigned char * d_inputArray, unsigned char * d_outputArray, int w, int h, int blurSize){

int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

if(Col<w && Row < h){
int pixVal = 0;
int pixels = 0;

for(int blurRow = -blurSize; blurRow < blurSize+1; ++blurRow){
for(int blurCol = -blurSize; blurCol < blurSize+1; ++blurCol){
int curRow = Row + blurRow;
int curCol = Col + blurCol;

//verify we have a valid image pixel
if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
pixVal += d_inputArray[curRow*w+curCol];
pixels++; // keep track of number of pixels in the avg
}
}
}

//write our new pixel value out
d_outputArray[Row*w+Col] = (unsigned char)(pixVal/pixels);


}

}