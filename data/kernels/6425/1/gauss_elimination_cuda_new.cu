#include "includes.h"
//==========================================================================================================
// A small snippet of code to solve equation of types Ax=B using Gaussian Elimniation
// Author - Anmol Gupta, Naved Ansari
// Course - EC513 - Introduction to Computer Architecture
// Boston University
//==========================================================================================================

//==========================================================================================================
// Command to compile the code
//nvcc -o GaussianElimination GaussianElimination.cu
//==========================================================================================================


// Assertion to check for errors
__global__ void gauss_elimination_cuda_new(float *a_d, float *b_d ,int size) {
int i, j;
int idy = threadIdx.x;

__shared__ float temp[MAXSIZE+10][MAXSIZE+10];
//copy to share
for(i=0; i<size+1; i++){
temp[idy][i] = a_d[(idy * (size+1)) + i];
//cuPrintf("T idy=%d, num = %d, temp=%f\n", idy, i, temp[idy][i]);
}
__syncthreads();


//loop through every row, calculate every column in parallel
for(i=1; i<size; i++){
//cuPrintf("\nthread %d(idy) going to loop %d(i)\n", idy, i);
if(idy >= i){
float t[MAXSIZE+10];
//perform calculation
for(j=0; j<size+1; j++){
if(j >= i-1){
t[j] = temp[i-1][j] - (temp[i-1][i-1] / temp[idy][i-1]) * temp[idy][j];
//cuPrintf("calculate No %d, answer %f\n", j, t);

}
}
__syncthreads();
//store data
for(j=0; j<size+1; j++){
if(j >= i-1){
temp[idy][j] = t[j];
}
}
}
__syncthreads();
}

//copy to host
for(i=0; i<size+1; i++){
b_d[idy * (size+1) + i] = temp[idy][i];
}
}