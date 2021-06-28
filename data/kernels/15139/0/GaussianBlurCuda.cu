#include "includes.h"
//iojpegparts.cu


__global__ void GaussianBlurCuda (unsigned char *pic, unsigned char * outpic, double *mask, int *size){ // size: width, height, mask_width
int pxPosCen = blockIdx.x * blockDim.x + threadIdx.x;
if (pxPosCen >= size[0]*size[1] || pxPosCen < 0) return;
int row, col, x, y, pos;
row = pxPosCen/size[0]; // pixel position taken as width major
col = pxPosCen%size[0];
double sumout[3];
sumout[0] = 0;
sumout[1] = 0;
sumout[2] = 0;
if (row < size[2]/2 || row >= (size[1] - (size[2]/2))) return;
if (col < size[2]/2 || col >= (size[0] - (size[2]/2))) return;
for (int i=0;i<size[2]*size[2];i++){
x = i%size[2] + col - size[2]/2;
y = i/size[2] + row - size[2]/2;
pos = (y*size[0]  + x)*3;
sumout[0]+=(double)(*(pic+pos  )) * mask[i];
sumout[1]+=(double)(*(pic+pos+1)) * mask[i];
sumout[2]+=(double)(*(pic+pos+2)) * mask[i];
}
pos = pxPosCen*3;
*(outpic+pos) = (unsigned char) sumout[0];
*(outpic+pos+1) = (unsigned char) sumout[1];
*(outpic+pos+2) = (unsigned char) sumout[2];
}