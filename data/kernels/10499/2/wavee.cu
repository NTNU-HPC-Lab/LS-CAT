#include "includes.h"

extern "C"
__global__ void wavee(int* tab, unsigned int rowSize, unsigned int centerX, unsigned int centerY, float A, float lambda, float time, float fi, unsigned int N)
{

int index = threadIdx.x + blockDim.x * blockIdx.x;
int w = int(index/rowSize);
int h = index%rowSize;

if ( w*rowSize+h < N ) {
float dx = 0;
if(centerX > w) {
dx = centerX - w;
} else {
dx = w - centerX;
}
float dy = 0;
if(centerY > h) {
dy = centerY - h;
} else {
dy = h - centerY;
}
float distance = pow(dx,2) + pow(dy,2);
distance = sqrt(distance);

float pi = 3.1415f;
float v = 1.0f;
float T = lambda/v;
float ww = 2.0f*pi/T;
float k = 2.0f*pi/lambda;
float f = A * sin( ww*time - k*distance + fi );

float res = f * 127 + 127;
tab[index] = int(res);

}


}