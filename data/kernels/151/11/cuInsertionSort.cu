#include "includes.h"
__global__ void cuInsertionSort(float *dist, int width, int pitch, int height, int k){

// Variables
int l,i,j;
float *p;
float v, max_value;
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

if (xIndex<width){

// Pointer shift and max value
p         = dist+xIndex;
max_value = *p;

// Part 1 : sort kth firt element
for (l=pitch;l<k*pitch;l+=pitch){
v = *(p+l);
if (v<max_value){
i=0; while (i<l && *(p+i)<=v) i+=pitch;
for (j=l;j>i;j-=pitch)
*(p+j) = *(p+j-pitch);
*(p+i) = v;
}
max_value = *(p+l);
}

// Part 2 : insert element in the k-th first lines
for (l=k*pitch;l<height*pitch;l+=pitch){
v = *(p+l);
if (v<max_value){
i=0; while (i<k*pitch && *(p+i)<=v) i+=pitch;
for (j=(k-1)*pitch;j>i;j-=pitch)
*(p+j) = *(p+j-pitch);
*(p+i) = v;
max_value  = *(p+(k-1)*pitch);
}
}
}
}