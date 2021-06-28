#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void interpolateMulti(int points, int noDims, int dimRes, float* surrogate, float* predictors, float* results) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < points) {
float *lower, *upper, *coeffs;
int *lowerInd;
lower = (float*)malloc((noDims)*sizeof(float));
upper = (float*)malloc((noDims)*sizeof(float));
coeffs = (float*)malloc(((int)pow(2,noDims-1))*sizeof(float));
lowerInd = (int*)malloc((noDims)*sizeof(float));

for (int jj = 0; jj < noDims; jj++) {
lower[jj] = surrogate[jj*dimRes];
upper[jj] = surrogate[(jj+1)*dimRes - 1];
lowerInd[jj] = (int)((dimRes-1)*(predictors[noDims*idx+jj] -
lower[jj])/(upper[jj] - lower[jj]));

if (lowerInd[jj] >= (dimRes-1)) {
lowerInd[jj] = dimRes-2;
} else if (lowerInd[jj] < 0){
lowerInd[jj] = 0;
}
}

// Let's interpolate
// Uppermost dimensions x value
float x0 = surrogate[lowerInd[0]];
float x1 = surrogate[lowerInd[0]+1];
float xd = (predictors[noDims*idx] - x0)/(x1-x0);

// First, assign the yvalues to the coefficients matrix
for (int jj = 0; jj < (int)pow(2,noDims-1); jj++) {
// Get the indices of the yvalues of the lower and upper bounding
// values on this dimension.
int idxL = dimRes*noDims;

for (int kk = 1; kk < noDims; kk++) {
int rem = ((int)(jj/((int)pow(2,noDims - kk - 1))) + 1) - 2*
(int)(((int)(jj/((int)pow(2,noDims - kk - 1))) + 1)/2);
if(rem > 0) {
idxL += lowerInd[kk]*(int)pow(dimRes,noDims - kk - 1);
} else {
idxL += (lowerInd[kk]+1)*(int)pow(dimRes,noDims - kk - 1);
}
}

int idxU = idxL + (lowerInd[0]+1)*(int)pow(dimRes,noDims-1);

idxL += lowerInd[0]*(int)pow(dimRes,noDims-1);

coeffs[jj] = surrogate[idxL]*(1 - xd) + surrogate[idxU]*xd;
}

// Now we work our way down the dimensions using our computed
// coefficients to get the interpolated value.
for (int jj = 1; jj < noDims; jj++) {
// Get the current dimension x value
x0 = surrogate[jj*dimRes + lowerInd[jj]];
x1 = surrogate[jj*dimRes + lowerInd[jj] + 1];
xd = (predictors[jj] - x0)/(x1-x0);

for (int kk = 0; kk < (int)pow(2,jj); kk++) {
int jump = (int)pow(2,noDims - jj - 2);
coeffs[kk] = coeffs[kk]*(1 - xd) + coeffs[kk + jump]*xd;
}
}

// Free variables
free(lowerInd);
free(coeffs);
free(upper);
free(lower);
// Output the result
results[idx] = coeffs[0];
}
}