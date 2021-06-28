#include "includes.h"
__global__ static void update_e(int objs,double* e,double* kval,double b_old,double b_new,int i,int j,int yi,int yj,double ai_old,double ai_new,double aj_old,double aj_new){
int id=blockDim.x * blockIdx.x + threadIdx.x;
if (id<objs){
double val=e[id];
val+=(b_new-b_old);
double ti=yi*kval[i*objs+id];
double tj=yj*kval[j*objs+id];
val += ti*(ai_new-ai_old);
val += tj*(aj_new-aj_old);
e[id]=val;
}
}