#include "includes.h"
__global__ void Blurrig(unsigned char* output, unsigned char* input, int height, int width) {
int Col = threadIdx.x + blockIdx.x * blockDim.x;
int Row = threadIdx.y + blockIdx.y * blockDim.y;

if (Col < width && Row < height) {
int pixVal = 0;
int pixels = 0;
for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
{
for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
{
int curRow = Row + blurRow;
int curCol = Col + blurCol;
//verify we have a valid image pixel
if (curRow > -1 && curRow<height && curCol>-1 && curCol < width) {
pixVal += input[curRow * width + curCol];
pixels++;//keep track of number of pixels in the avg
}
}
}
//write our new pixel value
output[Row * width + Col] = (unsigned char)(pixVal / pixels);
}
}