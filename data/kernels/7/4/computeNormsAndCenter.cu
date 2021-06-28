#include "includes.h"
__global__ void computeNormsAndCenter( float *norms, float *center, float *masses, float4 *positions, int *blocknums, int *blocksizes ) {
int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
float totalmass = 0.0;
for( int j = blocknums[blockNum]; j <= blocknums[blockNum] + blocksizes[blockNum] - 1; j += 3 ) {
float mass = masses[ j / 3 ];
center[blockNum * 3 + 0] = positions[j / 3].x * mass;
center[blockNum * 3 + 1] = positions[j / 3].y * mass;
center[blockNum * 3 + 2] = positions[j / 3].z * mass;
totalmass += mass;
}

norms[blockNum] = sqrt( totalmass );
center[blockNum * 3 + 0] /= totalmass;
center[blockNum * 3 + 1] /= totalmass;
center[blockNum * 3 + 2] /= totalmass;
}