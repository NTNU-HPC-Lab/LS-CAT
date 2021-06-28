#include "includes.h"

using namespace std;




__global__ void graphGenerate (float *a, float *b, int n){
int i= blockDim.x * blockIdx.x + threadIdx.x;

if (i<n){
a[i]=threadIdx.x*2;
b[i]=threadIdx.x;
}

}