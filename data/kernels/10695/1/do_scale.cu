#include "includes.h"
__global__ void do_scale(uint8_t * inBuffer, uint8_t * outBuffer, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight, double ratioHeight, double ratioWidth, double xRatio, double yRatio, uint32_t outHeightOffset, uint32_t outWidthOffset, uint32_t inStep, uint32_t outStep)
{
uint32_t outRowIndex=blockIdx.x+outHeightOffset;
uint32_t outColIndex=threadIdx.x+outWidthOffset;
uint32_t outIndex=(outRowIndex)*outWidth+outColIndex;

uint32_t inX00row=outRowIndex/ratioHeight;
uint32_t inX00col=outColIndex/ratioWidth;
//if (inX00row >= inHeight-2){
//printf("==edge inX00row[%u]inHeight[%u]==\n", inX00row, inHeight);
//inX00row=inHeight-2;
//}
//if (inX00col >= inWidth-2){
//printf("==edge inX00col[%u]inWidth[%u]==\n", inX00col, inWidth);
//inX00col=inWidth-2;
//}
uint8_t inX00=inBuffer[(inX00row*inWidth+inX00col)*inStep];
uint8_t inX01=inBuffer[(inX00row*inWidth+inX00col+1)*inStep];
uint8_t inX10=inBuffer[((inX00row+1)*inWidth+inX00col)*inStep];
uint8_t inX11=inBuffer[((inX00row+1)*inWidth+inX00col+1)*inStep];
outBuffer[outIndex*outStep]=yRatio*(xRatio*(inX00)+(1-xRatio)*(inX01))+(1-yRatio)*(xRatio*(inX10)+(1-xRatio)*(inX11));
}