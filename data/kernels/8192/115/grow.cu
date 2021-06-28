#include "includes.h"
__global__ void grow(float *matrices, const int dimension, const int coefficients, const int population, float *chromosomes, const float * noise, const float mutationRate, const int kept, const float* fitnesses, int *mark, const int alpha){

int i, wloc;

curandState st;

// For up to a 1D grid of 3D blocks...
int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

int chromOffset = threadGlobalID * coefficients;
int parent1, parent2, point;
float tmp1, tmp2;

// Init the random number generator
curand_init((int)noise[threadGlobalID] << threadGlobalID, threadGlobalID * (threadGlobalID == population - 1 ? noise[0] : noise[threadGlobalID]), 0, &st);

// Repopulate
// The threads with the keepmask are kept, all others are replaced with crossovers
if (threadGlobalID > kept - 1){
// pick two parents -- 0 is not included in the random distribution
parent1 = floor(curand_uniform(&st) * kept);
parent2 = floor(curand_uniform(&st) * kept);
//pick a point on the chromosome
point = floor(curand_uniform(&st) * coefficients);
for (i = 0; i < point; i++){
chromosomes[chromOffset + i] = chromosomes[parent1 * coefficients + i];
}
//Copy past the point for parent 2
for (i = point; i < coefficients; i++){
chromosomes[chromOffset + i] = chromosomes[parent2 * coefficients + i];
}
}

// Mutate children
if (threadGlobalID > kept - 1){
for (i = 0; i < coefficients; i++){
if (curand_uniform(&st) <= mutationRate){
if (curand_uniform(&st) < 0.5){
chromosomes[chromOffset + i] = curand_uniform(&st) * -1 * alpha;
}
else{
chromosomes[chromOffset + i] = curand_uniform(&st) * alpha;
}
}
}
}

// Permute
if (threadGlobalID < coefficients){
// Mark genes for permutation
for (i = 0; i < population; i++){
if (curand_uniform(&st) < (1 - sqrt((fitnesses[i] - fitnesses[population - 1]) / (fitnesses[0] - fitnesses[population - 1])))){
mark[coefficients * i + threadGlobalID] = 1;
}
else{
mark[coefficients * i + threadGlobalID] = 0;
}
}

wloc = -1;
// Permute selected genes
for (i = 0; i < population; i++){
if (mark[coefficients * i + threadGlobalID] == 1){
if (wloc == -1){
wloc = i;
tmp1 = chromosomes[coefficients * i + threadGlobalID];
}
else{
tmp2 = chromosomes[coefficients * i + threadGlobalID];
chromosomes[coefficients * i + threadGlobalID] = tmp1;
tmp1 = tmp2;
}
}
}
if (wloc != -1){
chromosomes[coefficients * wloc + threadGlobalID] = tmp1;
}
}

__syncthreads();
//Place into relevant matrix
for (i = 0; i < dimension*dimension; i++){
matrices[threadGlobalID * dimension * dimension + i] = 0.0f;
}
}