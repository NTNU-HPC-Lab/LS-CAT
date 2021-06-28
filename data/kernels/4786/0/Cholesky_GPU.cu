#include "includes.h"
using namespace std;

const int MAX = 100;



__global__ void Cholesky_GPU(double *matrix, int n){

//n threads running in parallel

//int x = blockIdx.x;
int y = threadIdx.x;
//int i = x;
int j = y;

extern __device__ __shared__ double localMatrix[];
//	extern __device__ __shared__ double sum[];
//matrix2d[x][y] = matrix1d[x*n+y]

//Copy to shared mem

for(int i=0; i<n; i++)
localMatrix[i*n+j] = matrix[i*n+j];

localMatrix[n*n+j] = 0; // sum column
__syncthreads();

//Do the calc;
#pragma unroll
for(int i=0; i<n; i++){
if(j<i){
localMatrix[i*n+j] = 0;
}
if(j>=i) {
localMatrix[n*n+j]=0;//initialize sum to 0
for(int k=0; k<i; k++)
localMatrix[n*n+j] +=localMatrix[k*n+i]*localMatrix[k*n+j]; // sums
//if(j<i){
//	localMatrix[i*n+j]=0;
//}
if(i == j){
localMatrix[i*n+j] = sqrt(localMatrix[i*n+j] - localMatrix[n*n+j]);
}if(j > i){
localMatrix[i*n+j] = (localMatrix[i*n+j] - localMatrix[n*n+j])/localMatrix[i*n+i];
}
}
}


__syncthreads();


for(int i=0; i<n; i++)
matrix[i*n+j] = localMatrix[i*n+j];
//Copy back



}