#include "includes.h"
__global__ void gpuIt3(float *tNew,float *tOld,float *tOrig,int x,int y,int z,float k,float st) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
if(i < x*y*z){

if(i == 0){ // front upper left corner
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 0;
}
else if(i == x-1){ // front upper right corner
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = .1;
}
else if(i == x*y-1){ // front lower right corner
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = .2;
}
else if(i == x*y-x){ // front lower left corner
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = .3;
}
else if(i == x*y*(z-1) ){ // back upper left corner
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = .4;
}
else if(i == x*y*(z-1) + x-1){ // back upper right corner
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = .5;
}
else if(i == x*y*z-1){ // back lower right corner
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = .6;
}
else if(i == x*y*z - x){ // back lower left corner
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = .7;
}

else if(i - x < 0){ // front top edge
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = .8;
}
else if(i%x == x-1 && i<x*y){ // front right edge
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = .9;
}
else if(i+x > x*y && i < (x*y)){ // front bottom edge
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1;
}
else if(i%x == 0 && i<x*y){ // front left edge
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 2;
}

else if(i > (x*y*z - x*y) && i < (x*y*z - (x-1)*y)){ // back top edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 3;
}
else if(i%x == x-1 && i > (x*y*(z-1))){ // back right edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = 4;
}
else if(i+x > x*y*z){ // back bottom edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 5;
}
else if(i%x == 0 && i > x*y*(z-1)){ // back left edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 6;
}

// the corner sides going front to back
else if(i%(x*y) == 0){ // upper left edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 7;
}
else if(i%(x*y) == x-1){ // upper right edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = 8;
}
else if(i%(x*y) == x*y-1){ // lower right edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = 9;
}
else if(i%(x*y) == x*y-x){ // lower left edge
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 9.1;
}

// else ifs here are vague because other options already completed
else if(i < x*y){ // front face
tNew[i] = tOld[i]+k*(tOld[i]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1.1;
}
else if(i > x*y*(z-1)){ // back face
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1.2;
}
else if(i%(x*y) < x){ // top face
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1.3;
}
else if(i%(x*y) > x*(y-1)){ // bottom face
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1.4;
}
else if(i%(x) == x-1){ // right face
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i]-6*tOld[i]);
//tNew[i] = 1.5;
}
else if(i%(x) == 0){ // left face
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i]+tOld[i+1]-6*tOld[i]);
//tNew[i] = 1.6;
}
else{ // all in the middle
//                       front        back         top       bottom     left     right
tNew[i] = tOld[i]+k*(tOld[i-(x*y)]+tOld[i+(x*y)]+tOld[i-x]+tOld[i+x]+tOld[i-1]+tOld[i+1]-6*tOld[i]);
}


//tNew[i] = i%(x*y);
// replace heaters
if(tOrig[i] != st){
tNew[i] = tOrig[i];
}

}
}