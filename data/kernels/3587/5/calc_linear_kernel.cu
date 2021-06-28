#include "includes.h"
__global__ static void calc_linear_kernel(int objs,int coords,double* x,double* out){
int id=blockDim.x * blockIdx.x + threadIdx.x;
int i=id/objs;
int j=id%objs;
if (i<objs){

double r=0.0;
for (int k=0;k<coords;k++){
r+=x[objs*k+i]*x[objs*k+j];
}
out[objs*i+j]=r;
}
}