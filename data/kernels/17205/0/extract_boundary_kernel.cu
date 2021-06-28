#include "includes.h"
using namespace std;
#define ITERATIONS 40000


enum pixel_position {INSIDE_MASK, BOUNDRY, OUTSIDE};

__global__ void extract_boundary_kernel(float *maskIn, int *boundryPixelArray, int source_nchannel, int source_width, int source_height){
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
for(int channel = 0; channel < source_nchannel; channel++){
if(x<source_width && y<source_height){
int id = x + source_width * y + source_width * source_height * channel;
if(x==0 && y==0 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x==0 && y==source_height-1 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x==source_width-1 && y==0 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x==source_width-1 && y==source_height-1 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x==0 && y < source_height-1 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x==source_width-1 && y < source_height-1 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x < source_width-1 && y==0 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else if(x < source_width-1 && y==source_height-1 && maskIn[id]){
boundryPixelArray[id]=OUTSIDE;
}
else{
int id_right = x+1 + y*source_width + channel * source_width * source_height;
int id_left = x-1 + y*source_width + channel * source_width * source_height;
int id_up = x + (y+1)*source_width + channel * source_width * source_height;
int id_down = x + (y-1)*source_width + channel * source_width * source_height;

if(maskIn[id]>=0.5&&maskIn[id_right]>=0.5&&maskIn[id_left]>=0.5&&maskIn[id_up]>=0.5&&maskIn[id_down]>=0.5){
boundryPixelArray[id] = INSIDE_MASK;
}
else if(maskIn[id]){
boundryPixelArray[id] = BOUNDRY;
}
else{
boundryPixelArray[id] = OUTSIDE;
}
}
}
}
}