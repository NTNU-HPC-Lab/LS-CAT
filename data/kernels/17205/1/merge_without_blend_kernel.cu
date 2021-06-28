#include "includes.h"
using namespace std;
#define ITERATIONS 40000


enum pixel_position {INSIDE_MASK, BOUNDRY, OUTSIDE};

__global__ void merge_without_blend_kernel(float *srcimg, float *targetimg, float *outimg, int *boundary_array,int source_nchannel, int source_width, int source_height){
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
for(int channel = 0; channel < source_nchannel; channel++){
int id = x + y*source_width + channel * source_width * source_height;
if(boundary_array[id] == INSIDE_MASK){
outimg[id] = targetimg[id];
}
else{
outimg[id] = srcimg[id];
}
}
}