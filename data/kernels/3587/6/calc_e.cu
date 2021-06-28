#include "includes.h"
__global__ static void calc_e(int objs,double* a,double b,int* y,double* kval,double* e){
int id=blockDim.x * blockIdx.x + threadIdx.x;
if (id<objs){
double fx=b;
for (int i=0;i<objs;i++){
//access to a and y are not coalesced
fx+=a[i]*y[i]*kval[i*objs+id];
}
e[id]=fx-y[id];
}
}