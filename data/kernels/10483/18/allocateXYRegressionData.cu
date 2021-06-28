#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void allocateXYRegressionData(int noPaths, int noControls, int noDims, int nYears, float* speciesParams, int year, int* controls, float* xin, float *condExp, int *dataPoints, float *xvals, float *yvals) {

for (int ii = 0; ii < noControls; ii++) {
dataPoints[ii] = 0;
}

//    // For each path
for (int ii = 0; ii < noPaths; ii++) {
if (controls[ii] >= noControls) {
printf("Invalid control %d\n",controls[ii]);
}

// NOT CHECKING VALIDITY
//        yvals[noPaths*controls[ii] + dataPoints[controls[ii]]] = condExp[(
//                year + 1)*noPaths + ii];

//        // Save the input dimension values to the corresponding data group
//        for (int jj = 0; jj < noDims; jj++) {
//            xvals[controls[ii]*noPaths*noDims + jj*noPaths + dataPoints[
//                    controls[ii]]] = xin[ii*noDims + jj];
//        }

////        printf("%6d | %3d: %6.0f %15.0f %15.0f\n",ii,controls[ii],xin[ii*noDims],
////                xin[ii*noDims + 1],yvals[noPaths*controls[ii] + dataPoints[controls[ii]]]);

//        // Increment the number of data points for this control
//        dataPoints[controls[ii]] += 1;

// CHECKING
// First check that the path is in-the-money. If it isn't we do not use
// it
bool valid = true;
for (int jj = 0; jj < (noDims-1); jj++) {
if (xin[ii*noDims + jj] < speciesParams[8*jj + 3]) {
valid = false;
break;
}
}

if (valid || controls[ii] == 0) {
// Save the conditional expectation
yvals[noPaths*controls[ii] + dataPoints[controls[ii]]] = condExp[(
year + 1)*noPaths + ii];

// Save the input dimension values to the corresponding data group
for (int jj = 0; jj < noDims; jj++) {
xvals[controls[ii]*noPaths*noDims + jj*noPaths + dataPoints[
controls[ii]]] = xin[ii*noDims + jj];
}

// Increment the number of data points for this control
dataPoints[controls[ii]] += 1;
}
}
}