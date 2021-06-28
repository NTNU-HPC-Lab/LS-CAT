#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void mteKernel(int noPaths, int nYears, int noPatches, float timeStep, float* rgr, float* brownians, float* jumpSizes, float* jumps, float* speciesParams, float *initPops, float* caps, float*mmm, int* rowIdx, int* elemsPerCol, float* pathPops, float* eps) {
// Global index for finding the thread number
int ii = blockIdx.x*blockDim.x + threadIdx.x;

// Only perform matrix multiplication sequentially for now. Later, if
// so desired, we can use dynamic parallelism because the card in the
// machine has CUDA compute capability 3.5
if (ii < noPaths) {
//extern __shared__ float s[];

// Initialise the prevailing population vector
for (int jj = 0; jj < noPatches; jj++) {
pathPops[(ii*2)*noPatches+jj] = initPops[jj];
}

float grMean = speciesParams[0];

for (int jj = 0; jj < nYears; jj++) {
// Movement and mortality. This component is very slow without
// using shared memory. As we do not know the size of the patches
// at compile time, we need to be careful how much shared memory we
// allocate. For safety, we assume that we will have less than
// 64KB worth of patch data in the mmm matrix. Using single
// precision floating point numbers, this means that we can only
// have up to 8,000 patches. As this number is extremely large, we
// set a limit outside this routine to have at most 300 patches.
for (int kk = 0; kk < noPatches; kk++) {
pathPops[(ii*2+1)*noPatches+kk] = 0.0;
}

int iterator = 0;
for (int kk = 0; kk < noPatches; kk++) {
for (int ll = 0; ll < elemsPerCol[kk]; ll++) {
pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
noPatches+rowIdx[iterator]]*mmm[iterator];
iterator++;
}
}

// UPDATE: NEED TO IMPLEMENT SHARED MEMORY AS WELL

// DEPRECATED - TO BE DELETED AT LATER STAGE
// Load the correct slice of the mmm matrix for each
// destination patch. Use the thread index as a helper to do
// this. Wait for all information to be loaded in before
// proceeding. We need to tile the mmm matrix here to obtain
// a sufficient speed up.

//            for (int kk = 0; kk < noTiles; kk++) {
//                int currDim = tileDim;

//                if (threadIdx.x < noPatches) {
//                    // First, allocate the memory for this tile
//                    if (kk == noTiles-1) {
//                        currDim = (int)(noTiles*tileDim == noPatches) ?
//                                (int)tileDim : (int)(noPatches - kk*tileDim);
//                    }

//                    for (int ll = 0; ll < currDim; ll++) {
//                        s[ll*noPatches + threadIdx.x] = mmm[kk*noPatches*
//                                tileDim + ll*noPatches + threadIdx.x];
//                    }
//                }
//                __syncthreads();

//                // Now increment the populations for this path
//                for (int kk = 0; kk < currDim; kk++) {
//                    for (int ll = 0; ll < noPatches; ll++) {
//                        pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
//                                noPatches+ll]*s[kk*noPatches + ll];
//                    }
//                }
//            }

//            for (int kk = 0; kk < noPatches; kk++) {
//                for (int ll = 0; ll < noPatches; ll++) {
////                    pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
////                            noPatches+ll]*s[ll];
//                    pathPops[(ii*2+1)*noPatches+kk] += pathPops[(ii*2)*
//                            noPatches+ll]*mmm[kk*noPatches+ll];
//                }
//            }

//            matrixMultiplicationKernel<<<noBlocks,noThreadsPerBlock>>>(pathPops
//                    + (ii*2)*noPatches, mmm, pathPops + (ii*2+1)*noPatches, 1,
//                    noPatches, noPatches);
//            cudaDeviceSynchronize();
//            __syncthreads();

// Natural birth and death

// Adjust the global growth rate mean for this species at this
// time step for this path.
float jump = (jumps[ii*nYears + jj] < speciesParams[6]) ? 1.0f :
0.0f;
float meanP = speciesParams[1];
float reversion = speciesParams[4];

float brownian = brownians[ii*nYears + jj]*speciesParams[2];
float jumpSize = jumpSizes[ii*nYears + jj]*pow(speciesParams[5],2)
- pow(speciesParams[5],2)/2;

grMean = grMean + reversion*(meanP - grMean)*timeStep + grMean
*brownian + (exp(jumpSize) - 1)*grMean*jump;

for (int kk = 0; kk < noPatches; kk++) {
float gr = speciesParams[7]*rgr[ii*(nYears*noPatches) + jj*
noPatches + kk]*grMean + grMean;
pathPops[(ii*2)*noPatches+kk] = pathPops[(ii*2+1)*noPatches+kk]
*(1.0f + gr*(caps[kk]-pathPops[(ii*2+1)*noPatches+kk])/
caps[kk]);
}
}

eps[ii] = 0.0f;
for (int jj = 0; jj < noPatches; jj++) {
eps[ii] += pathPops[(ii*2+1)*noPatches+jj];
}
}
}