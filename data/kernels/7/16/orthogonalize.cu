#include "includes.h"
__global__ void orthogonalize( float *eigvec, float *Qi_gdof, int cdof, int *blocksizes, int *blocknums, int largestblock ) {
int blockNum = blockIdx.x * blockDim.x + threadIdx.x;

// orthogonalize original eigenvectors against gdof
// number of evec that survive orthogonalization
int curr_evec = 6;
int size = blocksizes[blockNum];
int startatom = blocknums[blockNum] / 3;
for( int j = 0; j < size; j++ ) { // <-- vector we're orthogonalizing
// to match ProtoMol we only include size instead of size + cdof vectors
// Note: for every vector that is skipped due to a low norm,
// we add an additional vector to replace it, so we could actually
// use all size original eigenvectors
if( curr_evec == size ) {
break;
}

// orthogonalize original eigenvectors in order from smallest magnitude
// eigenvalue to biggest
// TMC The eigenvectors are sorted now
//int col = sortedPairs.at( j ).second;

// copy original vector to Qi_gdof -- updated in place
for( int l = 0; l < size; l++ ) {
//Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] = eigvec[blocknums[blockNum]+l][j];
Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] = eigvec[( blocknums[blockNum] + l ) * largestblock + j];
}

// get dot products with previous vectors
for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
// dot product between original vector and previously
// orthogonalized vectors
double dot_prod = 0.0;
for( int l = 0; l < size; l++ ) {
//dot_prod += Qi_gdof[blockNum*6*largestblock+l*6+k] * eigvec[blocknums[blockNum]+l][j];
dot_prod += Qi_gdof[blockNum * 6 * largestblock + l * 6 + k] * eigvec[( blocknums[blockNum] + l ) * largestblock + j];
}

// subtract from current vector -- update in place
for( int l = 0; l < size; l++ ) {
Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] = Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] - Qi_gdof[blockNum * 6 * largestblock + l * 6 + k] * dot_prod;
}
}

//normalize residual vector
double norm = 0.0;
for( int l = 0; l < size; l++ ) {
norm += Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] * Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec];
}

// if norm less than 1/20th of original
// continue on to next vector
// we don't update curr_evec so this vector
// will be overwritten
if( norm < 0.05 ) {
continue;
}

// scale vector
norm = sqrt( norm );
for( int l = 0; l < size; l++ ) {
Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] = Qi_gdof[blockNum * 6 * largestblock + l * 6 + curr_evec] / norm;
}

curr_evec++;
}

// 4. Copy eigenpairs to big array
//    This is necessary because we have to sort them, and determine
//    the cutoff eigenvalue for everybody.
// we assume curr_evec <= size
for( int j = 0; j < curr_evec; j++ ) {
//eval[startatom + j] = di[col]; No longer necessary

// orthogonalized eigenvectors already sorted by eigenvalue
for( int k = 0; k < size; k++ ) {
//eigvec[startatom + k][startatom + j] = Qi_gdof[blockNum*6*largestblock+k*6+j];
eigvec[( startatom + k )*largestblock + ( startatom + j )] = Qi_gdof[blockNum * 6 * largestblock + k * 6 + j];
}
}
}