#include "includes.h"
__global__ void  convolutionGPUkernel_1D(int *h_n, int *h_mascara,int *h_r,int n, int mascara){
int mitadMascara= (mascara/2);
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i<n){
int p=0;// almacena los valores temporales
int k= i - mitadMascara;
for (int j =0; j < mascara; j++){
if(k < n  && k >= 0){
p += h_n[k]*h_mascara[j];
}
else
p+=0;
k++;
}
h_r[i]=p;
}
}