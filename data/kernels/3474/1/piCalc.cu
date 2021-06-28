#include "includes.h"
__global__ void piCalc(double *area, double width, int rects) {
double mid, height;
// Get our index
int index = threadIdx.x + (blockIdx.x * blockDim.x);
// Pos in array
int id = index;
// do while we are inside our array
while(index<rects){
//Original pi algo
mid = (index + 0.5) * width;
height = 4.0 / (1.0 + mid * mid);
area[id] += height;
// Move our index
index += (blockDim.x*gridDim.x);
}
}