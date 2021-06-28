#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void printAverages(int nYears, int noSpecies, int noControls, int noPaths, float* totalPops, float* aars) {
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < nYears) {
float* totals, *aar;
totals = (float*)malloc(noSpecies*sizeof(float));

aar = (float*)malloc(noSpecies*noControls*sizeof(float));

for (int ii = 0; ii < noSpecies; ii++) {
totals[ii] = 0.0f;
for (int kk = 0; kk < noControls; kk++) {
aar[ii*noControls + kk] = 0;
}
}

for (int ii = 0; ii < noPaths; ii++) {
for (int jj = 0; jj < noSpecies; jj++) {
totals[jj] += totalPops[ii*noSpecies*(nYears+1) + (idx+1)*
noSpecies + jj];
for (int kk = 0; kk < noControls; kk++) {
aar[jj*noControls + kk] += aars[ii*(nYears+1)*noControls*
noSpecies + idx*noControls*noSpecies + jj*
noControls + kk];
}
}
}

for (int ii = 0; ii < noSpecies; ii++) {
totals[ii] = totals[ii]/(float)noPaths;
for (int jj = 0; jj < noControls; jj++) {
aar[ii*noControls + jj] = aar[ii*noControls + jj]/(float)
noPaths;
}
}

printf("Year: %d Total: %f C1: %f C2: %f C3: %f\n", idx,totals[0],aar[0],aar[1],aar[2]);

free(totals);
free(aar);
}
}