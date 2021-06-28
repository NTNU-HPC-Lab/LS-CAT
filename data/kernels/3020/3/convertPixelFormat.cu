#include "includes.h"
__global__ void convertPixelFormat(uint8_t* inputBgra, uint8_t* outputYuv, int numPixels){
int stride = gridDim.x * blockDim.x;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
short3 yuv16;
char3 yuv8;

while(idx<=numPixels){
if(idx<numPixels){
yuv16.x = 66*inputBgra[idx*4+2] + 129*inputBgra[idx*4+1] + 25*inputBgra[idx*4];
yuv16.y = -38*inputBgra[idx*4+2] + -74*inputBgra[idx*4+1] + 112*inputBgra[idx*4];
yuv16.z = 112*inputBgra[idx*4+2] + -94*inputBgra[idx*4+1] + -18*inputBgra[idx*4];

yuv8.x = (yuv16.x>>8)+16;
yuv8.y = (yuv16.y>>8)+128;
yuv8.z = (yuv16.z>>8)+128;

*(reinterpret_cast<char3*>(&outputYuv[idx*3])) = yuv8;
}
idx += stride;
}
}