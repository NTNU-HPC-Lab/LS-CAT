#include "includes.h"
__global__ void convolutionGlobal(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize){
size_t x = threadIdx.x + blockDim.x * blockIdx.x;
size_t y = threadIdx.y + blockDim.y * blockIdx.y;
size_t k = kernelSize;

int r=k/2;

//check for boundarys of the block
if(x>=w || y>=h) return;

//iterate over all channels
for(unsigned int c=0;c<nc;c++) {
float sum=0;
//do convolution
for(unsigned int i=0;i<k;i++){
unsigned int x_new;
//clamping x
if(x+r<i) x_new=0;
else if(x+r-i>=w) x_new=w-1;
else x_new=x+r-i;
for(unsigned int j=0;j<k;j++){
//clamping y
unsigned int y_new;
if(y+r<j)
y_new=0;
else if(y+r-j>=h)
y_new=h-1;
else
y_new=y+r-j;
sum+=kernel[i+j*k]*imgIn[x_new+y_new*w+w*h*c];
}
}
//imgOut[id(x,y,w,h,c)]=sum;
imgOut[x + y*w + c*w*h]=sum;
}
}