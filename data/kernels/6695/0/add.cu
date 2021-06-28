#include "includes.h"
#define N 10

//Sum Arrays
__global__ void add(int *x, int *y, int *z){
int tID = blockIdx.x;
if (tID < N){
z[tID] = x[tID] + y[tID];
}
}