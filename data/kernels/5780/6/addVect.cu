#include "includes.h"
__global__ void addVect(int *vect1 ,int *vect2 , int *resultVect){
int i = threadIdx.x + blockDim.x * blockIdx.x;
// printf("Thread id == %d || Block Id == %d\n",threadIdx.x,blockDim.x);
resultVect[i] = vect1[i] + vect2[i];
}