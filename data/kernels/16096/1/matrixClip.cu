#include "includes.h"
__global__ void matrixClip(double *a, double min, double max, double *c, int cr, int cc){

int x = blockIdx.x * blockDim.x + threadIdx.x; // col
int y = blockIdx.y * blockDim.y + threadIdx.y; // row


if(x < cc && y < cr){

if(a[y * cc + x] > max){
c[y * cc + x] = max;
}else{
if(a[y * cc + x] < min){
c[y * cc + x] = min;
}else{
c[y * cc + x] = a[y * cc + x];
}
}

}

}