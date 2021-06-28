#include "includes.h"



__global__ static void calc_predict(int objs,int objs_train,double* a,double b,int* y_train,double* kval,int* y){
int id=blockDim.x * blockIdx.x + threadIdx.x;
if (id<objs){
double fx=b;
for (int i=0;i<objs_train;i++){
//access to a and y are not coalesced
fx+=a[i]*y_train[i]*kval[i*objs+id];
}
y[id] = fx>=0 ? 1:-1;
}
}