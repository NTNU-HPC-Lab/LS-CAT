#include "includes.h"
__global__ void createHistCuda (float* siftCentroids, float* siftImage, int linesCent, int linesIm, float* temp)
{
__shared__ float cosines[BLOCK_SIZE][2];

size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
size_t idy = blockIdx.y;
size_t tid = threadIdx.x;

if(idx < linesCent){
int centin = idx * 128;
int imin = idy * 128;

//Cosine similarity code ------------
float sumab = 0;
float suma2 = 0;
float sumb2 = 0;

for(int k = 0; k < 128; k++){
sumab += siftCentroids[centin + k] * siftImage[imin + k];
suma2 += siftImage[imin + k] * siftImage[imin + k];
sumb2 += siftCentroids[centin + k] * siftCentroids[centin + k];
}

float cossim = sumab/(sqrtf(suma2)/sqrtf(sumb2));

//debug[idy*linesCent + idx] = cossim;
cosines[threadIdx.x][0] = cossim;
cosines[threadIdx.x][1] = idx;

__syncthreads();

for (unsigned int s=blockDim.x/2; s>0; s>>=1)
{
if (tid < s){
size_t tid2 = tid + s;
if(cosines[tid2][0] > cosines[tid][0]){
cosines[tid][0] = cosines[tid2][0];
cosines[tid][1] = cosines[tid2][1];
}
}
__syncthreads();
}

if (tid == 0){
temp[(blockIdx.y*gridDim.x + blockIdx.x)*2] = cosines[0][0];
temp[(blockIdx.y*gridDim.x + blockIdx.x)*2+1] = cosines[0][1];
}

}

}