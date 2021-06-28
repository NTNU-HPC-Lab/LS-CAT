#include "includes.h"
__global__ void geometricDOF( float *Qi_gdof, float4 *positions, float *masses, int *blocknums, int *blocksizes, int largestsize, float *norm, float *pos_center ) {
int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
for( int j = 0; j < blocksizes[blockNum] - 3; j += 3 ) {

int atom = ( blocknums[blockNum] + j ) / 3;
float mass = masses[atom];
float factor = sqrt( mass ) / norm[atom];

Qi_gdof[blockNum * largestsize * 6 + j * 6 + 0]   = factor;
Qi_gdof[blockNum * largestsize * 6 + ( j + 1 ) * 6 + 1] = factor;
Qi_gdof[blockNum * largestsize * 6 + ( j + 2 ) * 6 + 2] = factor;

float diff0 = positions[atom].x - pos_center[atom * 3 + 0];
float diff1 = positions[atom].y - pos_center[atom * 3 + 1];
float diff2 = positions[atom].z - pos_center[atom * 3 + 2];

Qi_gdof[blockNum * largestsize * 6 + ( j + 1 ) * 6 + 3] = diff2 * factor;
Qi_gdof[blockNum * largestsize * 6 + ( j + 2 ) * 6 + 3] = -diff1 * factor;

Qi_gdof[blockNum * largestsize * 6 + ( j ) * 6 + 4] = -diff2 * factor;
Qi_gdof[blockNum * largestsize * 6 + ( j + 2 ) * 6 + 4] = -diff0 * factor;

Qi_gdof[blockNum * largestsize * 6 + ( j ) * 6 + 5] = diff1 * factor;
Qi_gdof[blockNum * largestsize * 6 + ( j + 1 ) * 6 + 5] = -diff0 * factor;
}
// Normalize first vector
float rotnorm = 0.0;
for( int j = 0; j < blocksizes[blockNum]; j++ ) {
rotnorm += Qi_gdof[blockNum * largestsize * 6 + j * 6 + 3] * Qi_gdof[blockNum * largestsize * 6 + j * 6 + 3];
}

rotnorm = 1.0 / sqrt( rotnorm );

for( int j = 0; j < blocksizes[blockNum]; j++ ) {
Qi_gdof[blockNum * largestsize * 6 + j * 6 + 3] *= rotnorm;
}
}