#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__device__ void solveLinearSystem(int dims, float *A, float *B, float *C) {
// First generate upper triangular matrix for the augmented matrix
float *swapRow;
swapRow = (float*)malloc((dims+1)*sizeof(float));

for (int ii = 0; ii < dims; ii++) {
C[ii] = B[ii];
}

for (int ii = 0; ii < dims; ii++) {
// Search for maximum in this column
float maxElem = fabsf(A[ii*dims+ii]);
int maxRow = ii;

for (int jj = (ii+1); jj < dims; jj++) {
if (fabsf(A[ii*dims+jj] > maxElem)) {
maxElem = fabsf(A[ii*dims+jj]);
maxRow = jj;
}
}

// Swap maximum row with current row if needed
if (maxRow != ii) {
for (int jj = ii; jj < dims; jj++) {
swapRow[jj] = A[jj*dims+ii];
A[jj*dims+ii] = A[jj*dims+maxRow];
A[jj*dims+maxRow] = swapRow[jj];
}

swapRow[dims] = C[ii];
C[ii] = C[maxRow];
C[maxRow] = swapRow[dims];
}

// Make all rows below this one 0 in current column
for (int jj = (ii+1); jj < dims; jj++) {
float factor = -A[ii*dims+jj]/A[ii*dims+ii];

// Work across columns
for (int kk = ii; kk < dims; kk++) {
if (kk == ii) {
A[kk*dims+jj] = 0.0;
} else {
A[kk*dims+jj] += factor*A[kk*dims+ii];
}
}

// Results vector
C[jj] += factor*C[ii];
}
}
free(swapRow);

// Solve equation for an upper triangular matrix
for (int ii = dims-1; ii >= 0; ii--) {
C[ii] = C[ii]/A[ii*dims+ii];

for (int jj = ii-1; jj >= 0; jj--) {
C[jj] -= C[ii]*A[ii*dims+jj];
}
}
}
__global__ void multiLocLinReg(int noPoints, int noDims, int dimRes, int nYears, int noControls, int year, int control, int k, int* dataPoints, float *xvals, float *yvals, float *regression, float* xmins, float* xmaxes, float *dist, int *ind) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPoints) {
if (dataPoints[control] < 3) {
regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
+ control*(dimRes*noDims + (int)pow(dimRes,noDims)*2) + dimRes*
noDims + idx] = 0.0;
} else {
// First, deconstruct the index into the index along each dimension
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
xQ[ii] = ((float)dimIdx[ii])*(xmaxes[ii] - xmins[ii])/(float)(
dimRes - 1) + xmins[ii];
}

// 1. First find the k nearest neighbours to the query point (already)
// computed prior).

// 2. Build the matrices used in the calculation
// A - Input design matrix
// B - Input known matrix
// C - Output matrix of coefficients
float *A, *B, *X;

A = (float*)malloc(pow(noDims+1,2)*sizeof(float));
B = (float*)malloc((noDims+1)*sizeof(float));
X = (float*)malloc((noDims+1)*sizeof(float));

// Bandwidth for kernel
float h = dist[noPoints*(k-1) + idx];

for (int ii = 0; ii <= noDims; ii++) {
// We will use a kernel and normalise by the distance of
// the furthest point of the nearest k neighbours.

// Initialise values to zero
B[ii] = 0.0;

for (int kk = 0; kk < k; kk++) {
float d = dist[noPoints*kk + idx];
// Gaussian kernel (Not used for now)
float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);
// Epanechnikov kernel
//                    float z = 0.75*(1-pow(d/h,2));

if (ii == 0) {
B[ii] += yvals[ind[noPoints*kk + idx] - 1]*z;
} else {
B[ii] += yvals[ind[noPoints*kk + idx] - 1]*(xvals[(ind[noPoints
*kk + idx] - 1)*noDims + ii - 1] - xQ[ii-1])*z;
}
}

for (int jj = 0; jj <= noDims; jj++) {
A[jj*(noDims+1)+ii] = 0.0;

for (int kk = 0; kk < k; kk++) {
//                    float h = d_h[ind[kk]];
float d = dist[noPoints*kk + idx];
//                    For Gaussian kernel. Not used.
float z = exp(-(d/h)*(d/h)/2)/sqrt(2*M_PI);
//                        float z = 0.75*(1-pow(d/h,2));

if ((ii == 0) && (jj == 0)) {
A[jj*(noDims+1)+ii] += 1.0*z;
} else if (ii == 0) {
A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
)*noDims + jj - 1] - xQ[jj - 1])*z;
} else if (jj == 0) {
A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
)*noDims + ii - 1] - xQ[ii - 1])*z;
} else {
A[jj*(noDims+1)+ii] += (xvals[(ind[noPoints*kk + idx] - 1
)*noDims + jj - 1] - xQ[jj-1])*(xvals[(ind[
noPoints*kk + idx] - 1)*noDims + ii - 1] - xQ[ii
- 1])*z;
}
}
}
}

// Solve the linear system using LU decomposition.
solveLinearSystem(noDims+1,A,B,X);

// 4. Compute the y value at the x point of interest using the just-
//    found regression coefficients. This is simply the y intercept we
//    just computed and save to the regression matrix.
regression[year*noControls*(dimRes*noDims + (int)pow(dimRes,noDims)*2)
+ control*(dimRes*noDims + (int)pow(dimRes,noDims)*2) + dimRes*
noDims + idx] = /*yvals[ind[idx] - 1]*/ X[0];

// Free memory
free(A);
free(B);
free(X);
free(xQ);
free(dimIdx);
}
}
}