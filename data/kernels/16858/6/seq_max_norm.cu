#include "includes.h"
__global__ void seq_max_norm(float* mat1, int row, int col, float* norm){
*norm = 0;
for(int i=0; i<row; i++){
for(int j =0; j<col; j++)
*norm = max(abs(mat1[i*col+j]), *norm);
}
}