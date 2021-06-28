#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void rovCorrection(int noPoints, int noDims, int dimRes, int nYears, int noControls, int year, int control, float* regression) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPoints) {
float currVal = regression[year*noControls*(dimRes*noDims +
(int)pow(dimRes,noDims)*2) + control*(dimRes*noDims +
(int)pow(dimRes,noDims)*2) + dimRes*noDims + idx];

// The surrogate value cannot be greater than zero by definition
if (currVal > 0) {
regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
noDims)*2) + dimRes*noDims + idx] = 0.0;
}
}
}