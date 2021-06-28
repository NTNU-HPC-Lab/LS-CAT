#include "includes.h"
__global__ void matchHistCuda(float*qSet, float*dbSet, size_t qSize, size_t dbSize, size_t hSize, float*out){
size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
size_t idy = blockIdx.y*blockDim.y + threadIdx.y;

if(idx < qSize && idy < dbSize){
size_t qi = idx*hSize;
size_t dbi = idy*hSize;

//Cosine similarity code ------------
float sumab = 0;
float suma2 = 0;
float sumb2 = 0;

for(int k = 0; k < hSize; k++){
sumab += qSet[qi+k] * dbSet[dbi+k];
suma2 += qSet[qi+k] * qSet[qi+k];
sumb2 += dbSet[dbi+k] * dbSet[dbi+k];
}

float cossim = sumab/(sqrtf(suma2)/sqrtf(sumb2));
out[idy*qSize + idx] = cossim;
}
}