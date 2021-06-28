#include "includes.h"
__global__ void cuSearchDoublet( const int* nSpM, const float* spMmat, const int* nSpB, const float* spBmat, const int* nSpT, const float* spTmat, const float* deltaRMin, const float* deltaRMax, const float* cotThetaMax, const float* collisionRegionMin, const float* collisionRegionMax, int* nSpMcomp, int* nSpBcompPerSpM_Max, int* nSpTcompPerSpM_Max, int* nSpBcompPerSpM, int* nSpTcompPerSpM, int* McompIndex, int* BcompIndex, int* tmpBcompIndex, int* TcompIndex, int* tmpTcompIndex) {
extern __shared__ float sharedMem[];
int* mPos = (int*)sharedMem;
int* isMcompat = (int*)&mPos[1];

if (threadIdx.x == 0) {
*isMcompat = false;
}
__syncthreads();

float rM = spMmat[blockIdx.x + (*nSpM) * 3];
float zM = spMmat[blockIdx.x + (*nSpM) * 2];

bool isBcompat(true);
bool isTcompat(true);

int offset(0);

while (offset < max(*nSpB, *nSpT)) {
isBcompat = true;

// Doublet search for bottom hits
if (threadIdx.x + offset < *nSpB) {
float rB = spBmat[threadIdx.x + offset + (*nSpB) * 3];
float zB = spBmat[threadIdx.x + offset + (*nSpB) * 2];

float deltaR = rM - rB;
if (deltaR > *deltaRMax) {
isBcompat = false;
}

if (deltaR < *deltaRMin) {
isBcompat = false;
}

float cotTheta = (zM - zB) / deltaR;
if (fabsf(cotTheta) > *cotThetaMax) {
isBcompat = false;
}

float zOrigin = zM - rM * cotTheta;
if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax) {
isBcompat = false;
}

if (isBcompat == true) {
int bPos = atomicAdd(&nSpBcompPerSpM[blockIdx.x], 1);
tmpBcompIndex[bPos + (*nSpB) * blockIdx.x] = threadIdx.x + offset;
}
}

isTcompat = true;

// Doublet search for top hits
if (threadIdx.x + offset < *nSpT) {
float rT = spTmat[threadIdx.x + offset + (*nSpT) * 3];
float zT = spTmat[threadIdx.x + offset + (*nSpT) * 2];
float deltaR = rT - rM;
if (deltaR < *deltaRMin) {
isTcompat = false;
}

if (deltaR > *deltaRMax) {
isTcompat = false;
}

if (isTcompat == true) {
float cotTheta = (zT - zM) / deltaR;
if (fabsf(cotTheta) > *cotThetaMax) {
isTcompat = false;
}

float zOrigin = zM - rM * cotTheta;
if (zOrigin < *collisionRegionMin || zOrigin > *collisionRegionMax) {
isTcompat = false;
}
}

if (isTcompat == true) {
int tPos = atomicAdd(&nSpTcompPerSpM[blockIdx.x], 1);
tmpTcompIndex[tPos + (*nSpT) * blockIdx.x] = threadIdx.x + offset;
}
}

offset += blockDim.x;
}

__syncthreads();

if (threadIdx.x == 0) {
if (nSpBcompPerSpM[blockIdx.x] > 0 && nSpTcompPerSpM[blockIdx.x] > 0) {
*mPos = atomicAdd(nSpMcomp, 1);
*isMcompat = true;
McompIndex[*mPos] = blockIdx.x;

int bMax = atomicMax(nSpBcompPerSpM_Max, nSpBcompPerSpM[blockIdx.x]);
int tMax = atomicMax(nSpTcompPerSpM_Max, nSpTcompPerSpM[blockIdx.x]);
}
}

__syncthreads();

if (*isMcompat == true) {
offset = 0;
while (offset <
max(nSpBcompPerSpM[blockIdx.x], nSpTcompPerSpM[blockIdx.x])) {
if (threadIdx.x + offset < nSpBcompPerSpM[blockIdx.x]) {
BcompIndex[threadIdx.x + offset + (*nSpB) * (*mPos)] =
tmpBcompIndex[threadIdx.x + offset + (*nSpB) * blockIdx.x];
}

if (threadIdx.x + offset < nSpTcompPerSpM[blockIdx.x]) {
TcompIndex[threadIdx.x + offset + (*nSpT) * (*mPos)] =
tmpTcompIndex[threadIdx.x + offset + (*nSpT) * blockIdx.x];
}
offset += blockDim.x;
}
}
}