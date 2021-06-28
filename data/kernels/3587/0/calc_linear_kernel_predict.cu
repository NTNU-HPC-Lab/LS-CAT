#include "includes.h"



__global__ static void calc_linear_kernel_predict(int objs,int coords,double* x,int objs_train,double* x_train,double* out){
int id=blockDim.x * blockIdx.x + threadIdx.x;
int i=id/objs;
int j=id%objs;
if (i<objs_train){
double r=1.0;
for (int k=0;k<coords;k++){
r += x_train[coords*i+k] * x[coords*j+k];
}
out[id]=r;
}
}