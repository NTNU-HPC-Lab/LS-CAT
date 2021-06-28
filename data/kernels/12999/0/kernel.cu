#include "includes.h"
//*************inclución de librerias***************


//************variables globales***************

int N=93, dimx=1920, dimy=2560, tam_imag=1920*2560;

//**********KERNEL**************

float *leerMatrizVarianza(int d);

//*****************función main**********************

__global__ void kernel (float *max, float *var, int *top, int k){
int idx=threadIdx.x + blockIdx.x*blockDim.x;
int tam_imag=1920*2560;

if(idx<tam_imag){
if(var[idx]>max[idx]){
top[idx]=k;
max[idx]=var[idx];
}
}
}