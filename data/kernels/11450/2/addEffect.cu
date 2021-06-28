#include "includes.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

__device__ int getPosition(int x, int y, int width, int margin, int pixelPosition)
{
return (x + (y * width)) * margin + pixelPosition;
}
__global__ void addEffect( unsigned char* output_img, unsigned char* input_img, int width, int height, int nbBlocks)
{
int lengthY = (int)(height/nbBlocks)+1;
int startY = blockIdx.x * lengthY;
int endY = blockIdx.x * lengthY + lengthY;

if( endY > height )
endY = height;

int lengthX = (int)(width/blockDim.x)+1;
int startX = threadIdx.x * lengthX;
int endX = threadIdx.x * lengthX + lengthX;

if( endX > width )
endX = width;

for( int x = startX; x < endX; x++ )
{
for( int y = startY; y < endY; y++ )
{
int currentIndex = getPosition(x, y, width, 3, 0);
if( (input_img[currentIndex] + input_img[currentIndex+1] + input_img[currentIndex+2])/3 < 20)
{
output_img[currentIndex] = input_img[currentIndex];
output_img[currentIndex+1] = input_img[currentIndex+1];
output_img[currentIndex+2] = input_img[currentIndex+2];

for( int i = -4; i <= 4; i++ )
{
for( int j = -4; j <= 4; j++ )
{
if( x+i < 0 || x+i > width || y+j < 0 || y+j > height )
continue;

int neighbourIndex = getPosition( x+i, y+j, width, 3, 0);

if( neighbourIndex < 0 || neighbourIndex + 2 > width*height*3)
continue;

output_img[neighbourIndex] = 0;
output_img[neighbourIndex+1] = 0;
output_img[neighbourIndex+2] = 0;
}

}
}
}

}

}