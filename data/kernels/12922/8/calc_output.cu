#include "includes.h"
__global__ void calc_output(unsigned char * img_out, unsigned char * img_in, int * lut, int img_size){
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
const int gridW = gridDim.x * blockDim.x;
int img_position1 = iy * gridW + ix; //thesh mesa sthn eikona synarthsh tou gridW
__syncthreads();
if(img_position1 < img_size){
if(lut[img_in[img_position1]] > 255){
img_out[img_position1] = 255;
}
else{
img_out[img_position1] = (unsigned char)lut[img_in[img_position1]];
}
}


}