#include "includes.h"
__global__ void blur(unsigned char *pixels, int rows, int cols, int channels, int kernel, int numThreads){
int id = blockDim.x * blockIdx.x + threadIdx.x ;
int i = rows * id / numThreads;
int end = ( id == numThreads-1)? rows: rows * (id + 1) / numThreads;

int k = kernel/2;

for(; i<end; i++){
for (int j=0; j<cols; j++){
unsigned int blue=0.0, red=0.0, green=0.0;
double sum = 0.0;
for(int x=i-k; x<=i+k; x++){
for(int y=j-k; y<=j+k; y++){
if(x<rows && x>=0 && y<cols && y>=0){
sum += 1;
blue  += pixels[ (cols*x*channels+y*channels)+0];
green += pixels[ (cols*x*channels+y*channels)+1];
red   += pixels[ (cols*x*channels+y*channels)+2];
}
}
}
pixels[ (cols*i*channels+j*channels)+0] = (unsigned int) blue/sum;
pixels[ (cols*i*channels+j*channels)+1] = (unsigned int) green/sum;
pixels[ (cols*i*channels+j*channels)+2] = (unsigned int) red/sum;
}
}
}