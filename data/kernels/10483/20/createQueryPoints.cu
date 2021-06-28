#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void createQueryPoints(int noPoints, int noDims, int dimRes, int control, int noControls, int year, float* xmins, float* xmaxes, float* regression, float* queryPts) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPoints) {

// First, deconstruct the index into the index along each dimension
int *dimIdx;
dimIdx = (int*)malloc(noDims*sizeof(int));

int rem = idx;

for (int ii = 0; ii < noDims; ii++) {
int div = (int)(rem/pow(dimRes,noDims-ii-1));
dimIdx[ii] = div;
rem = rem - div*pow(dimRes,noDims-ii-1);
}

// We use the highest and lowest x values for each dimension
// among ALL the controls, not just for this control

// Get the query point coordinates
for (int ii = 0; ii < noDims; ii++) {
//            queryPts[idx + ii*noPoints] = ((float)dimIdx[ii])*(xmaxes[
//                    control*noDims + ii] - xmins[control*noDims + ii])/(
//                    float)(dimRes-1) + xmins[control*noDims + ii];
queryPts[idx + ii*noPoints] = ((float)dimIdx[ii])*(xmaxes[
noControls*noDims + ii] - xmins[noControls*noDims +
ii])/(float)(dimRes-1) + xmins[noControls*noDims +
ii];

// Save the X value for the query point
regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,
noDims)*2) + control*(dimRes*noDims + (int)pow(dimRes,
noDims)*2) + ii*dimRes + dimIdx[ii]] = queryPts[idx + ii*
noPoints];
}

free(dimIdx);
}
}