#include "includes.h"
__global__ void kernel0(int n, float a, float *x, float *y){

int i = blockIdx.x*blockDim.x + threadIdx.x;



//comment out this for-loop and uncomment the code in the main function for getting correct results
for (int i = 0; i < n; i++) {
x[i] = 1.0f;
y[i] = 2.0f;
}

if (i < n){
y[i] = a*x[i] + y[i];
}
}