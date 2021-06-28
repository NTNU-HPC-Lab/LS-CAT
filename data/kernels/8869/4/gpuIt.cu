#include "includes.h"
__global__ void gpuIt(float *tNew,float *tOld,float *tOrig,int x,int y,int z,float k,float st) {

int i = threadIdx.x + blockIdx.x * blockDim.x;
// may want an if(i < x*y*z) to prevent overflowing, likea thisa
if(i < x*y*z){

if(i == 0){ // top left corner
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 1;
}
else if(i == x-1){ // top right corner
tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 3;
}
else if(i == x*y - 1){ // bottom right corner
tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i-x] + tOld[i] - 4*tOld[i]);
//tNew[i] = 5;
}
else if(i == x*y - x){ // bottom left corner
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i-x] + tOld[i] - 4*tOld[i]);
//tNew[i] = 7;
}
else if(i%x == 0){ // left side
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 8;
}
else if(i%x == x-1){ // right side
tNew[i] = tOld[i] + k*(tOld[i] + tOld[i-1] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 4;
}
else if(i - x < 0){ // top row
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 2;
}
else if(i + x > x*y){ // bottom row
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i-x] + tOld[i] - 4*tOld[i]);
//tNew[i] = 6;
}
else{
tNew[i] = tOld[i] + k*(tOld[i+1] + tOld[i-1] + tOld[i-x] + tOld[i+x] - 4*tOld[i]);
//tNew[i] = 9;
}
//tNew[i] = i; // for debugging
// replace heaters
if(tOrig[i] != st){
tNew[i] = tOrig[i];
}
//tNew[i] = i%x;
}
}