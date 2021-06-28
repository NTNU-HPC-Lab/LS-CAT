#include "includes.h"
__global__ void hsv2rgb(float *inputH, float *inputS, float *inputV, uchar3 *output, int width, int height) {


int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int tid = y*width + x;

if (x<width){

if (y<height){

float H = inputH[tid];

float S = inputS[tid];

float V = inputV[tid];

float d =inputH[tid]/60;

int hi = (int)d%6;

float f = d - hi;

float l = V * (1 - S);

float m = V * (1 - f*S);

float n = V * (1 - (1 - f)*S);


if ((H>=0)&&(H<60)){

output[tid].x = (int)(V*255);
output[tid].y =(int) (n*255);
output[tid].z = (int)(l*255);

}
if ((H>=60)&&(H<120)){

output[tid].x = (int)(m*255);
output[tid].y = (int)(V*255);
output[tid].z = (int)(l*255);

}
if ((H>=120)&&(H<180)){

output[tid].x = (int)(l*255);
output[tid].y = (int)(V*255);
output[tid].z = (int)(n*255);

}
if ((H>=180)&&(H<240)){

output[tid].x = (int)(l*255);
output[tid].y = (int)(m*255);
output[tid].z = (int)(V*255);

}
if ((H>=240)&&(H<300)){

output[tid].x = (int)(n*255);
output[tid].y = (int)(l*255);
output[tid].z = (int)(V*255);

}

if ((H>=300)&&(H<360)){

output[tid].x = (int)(V*255);
output[tid].y = (int)(l*255);
output[tid].z = (int)(m*255);

}


}
}
}