#include "includes.h"
__device__ int getGlobalIdx_2D_2D()
{
int blockId = blockIdx.x
+ blockIdx.y * gridDim.x;

int threadId = blockId * (blockDim.x * blockDim.y)
+ (threadIdx.y * blockDim.x)
+ threadIdx.x;

return threadId;
}
__global__ void CudaTransform(unsigned char* dev_img, unsigned int *dev_accu, int w, int h){


//calculate index which this thread have to process
unsigned int index = getGlobalIdx_2D_2D();

//check index is in image bounds
if(index < (w*h)){
//calculate params
float hough_h = ((sqrt(2.0) * (float)(h>w?h:w)) / 2.0);

float center_x = w/2;
float center_y = h/2;

//calculate coordinates for corresponding index in entire image
int x = index % w;
int y = index / w;

if( dev_img[index] > 250 ){ //se il punto è bianco (val in scala di grigio > 250)
for(int t=0;t<180;t++){ //plot dello spazio dei parametri da 0° a 180° (sist. polare)

float r = ( ((float)x - center_x) * cos((float)t * DEG2RAD)) + (((float)y - center_y) * sin((float)t * DEG2RAD));

//dev_accu[ (int)((round(r + hough_h) * 180.0)) + t]++;
atomicAdd(&(dev_accu[ (int)((round(r + hough_h) * 180.0)) + t]), 1);

}
}
}

}