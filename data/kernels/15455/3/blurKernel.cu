#include "includes.h"
__global__ void blurKernel(uchar3 *in, uchar3 *out, int w, int h)
{
int Col = blockIdx.x*blockDim.x + threadIdx.x;
int Row = blockIdx.y*blockDim.y + threadIdx.y;

if(Col<w && Row<h)
{
int pixVal1 = 0;
// int pixVal2 = 0;
// int pixVal3 = 0;
int	pixels1 = 0;
// int pixels2 = 0;
// int pixels3 = 0;

for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1;++blurRow){
for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1;++blurCol)
{
int curRow = Row + blurRow;
int curCol = Col + blurCol;

if(curRow>-1 && curRow<h && curCol>-1 && curCol<w){
pixVal1+=static_cast<int>(in[curRow*w + curCol].x);
pixels1++;
pixVal1+=static_cast<int>(in[curRow*w + curCol].y);
pixels1++;
pixVal1+=static_cast<int>(in[curRow*w + curCol].z);
pixels1++;

}
}

}

out[Row*w+Col].x= static_cast<unsigned char>(pixVal1/pixels1);
out[Row*w+Col].y= static_cast<unsigned char>(pixVal1/pixels1);
out[Row*w+Col].z= static_cast<unsigned char>(pixVal1/pixels1);

}
}