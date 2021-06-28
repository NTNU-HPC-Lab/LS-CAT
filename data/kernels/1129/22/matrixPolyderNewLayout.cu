#include "includes.h"
__global__ void matrixPolyderNewLayout(const float *coefImg, float *coefImgDer, const int w, const int h, const int m, size_t yOffset){
size_t x = threadIdx.x + blockDim.x * blockIdx.x;
size_t y = threadIdx.y + blockDim.y * blockIdx.y;
if(x >= w || y >= h) return;

size_t xOffsetDer = m-1;
size_t yOffsetDer = w*xOffsetDer;

size_t xOffsetCoef = m;
size_t yOffsetCoef = w*xOffsetCoef;

for (int i = 0; i < m - 1; ++i) //if of degree d=2, we have n=3 coeffs ax'2 + bx +c
{
size_t idxDer = x*xOffsetDer + y*yOffsetDer + i;
size_t idxCoef = x*xOffsetCoef + y*yOffsetCoef + i;

coefImgDer[idxDer]=coefImg[idxCoef]*(m-i-1);
}
}