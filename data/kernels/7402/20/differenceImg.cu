#include "includes.h"
__global__ void differenceImg(float *d_Octave0,float *d_Octave1,float *d_diffOctave,int pitch,int height){

int x = blockIdx.x*blockDim.x+threadIdx.x;
int y = blockIdx.y*blockDim.y+threadIdx.y;

int index = y * pitch + x;
if(y<height)
d_diffOctave[index] = (d_Octave1[index] - d_Octave0[index]);

}