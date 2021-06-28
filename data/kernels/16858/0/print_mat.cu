#include "includes.h"
__global__ void print_mat(float* mat, int row, int col){
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id==0){
for(int i=0; i<row; i++){
for(int j =0; j<col; j++)
printf("%0.3f\t", mat[i*col+j]);
printf("\n");
}
printf("\n");
}
}