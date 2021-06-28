#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void buildGlobalQuadReg(int noPoints, int noDims, int dimRes, int nYears, int noControls, int year, int control, float* regCoeffs, float* xmins, float* xmaxes, float* regression) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPoints) {
// First deconstruct the index into the index along each dimension
int *dimIdx;
dimIdx = (int*)malloc(noDims*sizeof(int));

int rem = idx;

for (int ii = 0; ii < noDims; ii++) {
int div = (int)(rem/pow(dimRes,noDims-ii-1));
dimIdx[ii] = div;
rem = rem - div*pow(dimRes,noDims-ii-1);
}

// Get the query point coordinates
float *xQ;
xQ = (float*)malloc(noDims*sizeof(float));

for (int ii = 0; ii < noDims; ii++) {
xQ[ii] = ((float)dimIdx[ii])*(xmaxes[control*noDims + ii] -
xmins[control*noDims + ii])/(float)dimRes +
xmins[control*noDims + ii];
}

// Use the regression coefficients to compute the value at this query
// point
// Constant
float computed = regCoeffs[0];

// Linear Terms
for (int ii = 0; ii < noDims; ii++) {
computed += xQ[ii]*regCoeffs[ii+1];
}

// Quadratic and Interacting Terms
int counter = 0;
for (int ii = 0; ii < noDims; ii++) {
for (int jj = ii; jj < noDims; jj++) {
computed += xQ[ii]*xQ[jj]*regCoeffs[counter+1+noDims];
counter++;
}
}

// We know that a payoff cannot be greater than zero, so we adjust all
// conditional payoffs greater than zero to be zero.

if (computed >= 0.0) {
computed = 0.0;
}

regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
+ control*(dimRes*noDims + (int)pow(dimRes,noDims)*2) + dimRes*
noDims + idx] = computed;

// Free memory
free(xQ);
free(dimIdx);
}
}