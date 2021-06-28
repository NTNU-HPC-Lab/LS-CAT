#include "includes.h"




__global__ void oddeven(int* x,int I,int n)
{
int id=blockIdx.x;
if(I==0 && ((id*2+1)< n)){
if(x[id*2]>x[id*2+1]){
int X=x[id*2];
x[id*2]=x[id*2+1];
x[id*2+1]=X;
}
}
if(I==1 && ((id*2+2)< n)){
if(x[id*2+1]>x[id*2+2]){
int X=x[id*2+1];
x[id*2+1]=x[id*2+2];
x[id*2+2]=X;
}
}
}