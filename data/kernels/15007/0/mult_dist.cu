#include "includes.h"
#define H 5
#define W 5

using namespace std;

__global__ void mult_dist(int *d_A, int *d_B,int *d_C){
int i = blockIdx.y*blockDim.y+threadIdx.y;//todos los valores fila
int j = blockIdx.x*blockDim.x+threadIdx.x;//todos los valores columna
if(i < H && j < W){
int Pvalue = 0;
for(int k=0; k<H; k++){
Pvalue += d_A[i*W+k] * d_B[k*W+j];
}
d_C[i*W+j] = Pvalue;
}
}