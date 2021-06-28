#include "includes.h"
__global__ void orthogonalize23( float *Qi_gdof, int *blocksizes, int numblocks, int largestblock ) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
for( int j = 4; j < 6; j++ ) {
for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
float dot_prod = 0.0;
for( int l = 0; l < blocksizes[i]; l++ ) {
dot_prod += Qi_gdof[i * 6 * largestblock + l * 6 + k] * Qi_gdof[i * 6 * largestblock + l * 6 + j];
}
//dot_prod += Qi_gdof[i][l][k] * Qi_gdof[i][l][j];
for( int l = 0; l < blocksizes[i]; l++ ) {
Qi_gdof[i * 6 * largestblock + l * 6 + j] -= Qi_gdof[i * 6 * largestblock + l * 6 + k] * dot_prod;
}
//Qi_gdof[i][l][j] -= Qi_gdof[i][l][k] * dot_prod;
}

float rotnorm = 0.0;
for( int l = 0; l < blocksizes[i]; l++ ) {
rotnorm += Qi_gdof[i * 6 * largestblock + l * 6 + j] * Qi_gdof[i * 6 * largestblock + l * 6 + j];
}
//rotnorm += Qi_gdof[i][l][j] * Qi_gdof[i][l][j];

rotnorm = 1.0 / sqrt( rotnorm );

for( int l = 0; l < blocksizes[i]; l++ ) {
Qi_gdof[i * 6 * largestblock + l * 6 + j] *= rotnorm;
}
//Qi_gdof[i][l][j] *= rotnorm;
}
}