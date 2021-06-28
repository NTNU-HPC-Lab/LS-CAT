#include "includes.h"
__global__ void ker_gkylCartFieldAccumulateOffset(unsigned sInp, unsigned sOut, unsigned nCells, unsigned compStart, unsigned nCompInp, unsigned nCompOut, double fact, const double *inp, double *out) {
if (nCompInp < nCompOut) {
for (unsigned i=blockIdx.x*blockDim.x + threadIdx.x; i<nCells; i += blockDim.x * gridDim.x) {
for (unsigned c=0; c<nCompInp; ++c) {
out[sOut + i*nCompOut + compStart + c] += fact*inp[sInp + i*nCompInp + c];
}
}
}
else {
for (unsigned i=blockIdx.x*blockDim.x + threadIdx.x; i<nCells; i += blockDim.x * gridDim.x) {
for (unsigned c=0; c<nCompOut; ++c) {
out[sOut + i*nCompOut + c] += fact*inp[sInp + i*nCompInp + compStart + c];
}
}
}
}