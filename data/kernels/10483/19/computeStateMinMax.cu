#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void computeStateMinMax(int noControls, int noDims, int noPaths, int* dataPoints, float* xvals, float* xmins, float* xmaxes) {

for (int ii = 0; ii < noControls; ii++) {
float *xmin, *xmax;
xmin = (float*)malloc(noDims*sizeof(float));
xmax = (float*)malloc(noDims*sizeof(float));

if (ii == 0 || dataPoints[ii] > (noDims+1)) {
for (int jj = 0; jj < noDims; jj++) {
xmin[jj] = xvals[ii*noDims*noPaths + jj*noPaths];
xmax[jj] = xmin[jj];
}

for (int jj = 0; jj < noDims; jj++) {
for (int kk = 0; kk < dataPoints[ii]; kk++) {
float xtemp = xvals[ii*noDims*noPaths + jj*noPaths + kk];
if (xmin[jj] > xtemp) {
xmin[jj] = xtemp;
} else if (xmax[jj] < xtemp) {
xmax[jj] = xtemp;
}
}
}

//        for (int jj = 0; jj < noDims; jj++) {
//            xmin[jj] = xvals[ii*noDims*noPaths + jj];
//            xmax[jj] = xmin[jj];
//        }

//        for (int jj = 0; jj < dataPoints[ii]; jj++) {
//            for (int kk = 0; kk < noDims; kk ++) {
//                float xtemp = xvals[ii*noDims*noPaths + jj*noDims + kk];
//                if (xmin[kk] > xtemp) {
//                    xmin[kk] = xtemp;
//                } else if (xmax[kk] < xtemp) {
//                    xmax[kk] = xtemp;
//                }
//            }
//        }

for (int jj = 0; jj < noDims; jj++) {
xmins[ii*noDims + jj] = xmin[jj];
xmaxes[ii*noDims + jj] = xmax[jj];
//            printf("Control %d: Xmin = %f Xmax = %f\n",ii,xmin[jj],xmax[jj]);
}
} else {
for (int jj = 0; jj < noDims; jj++) {
xmins[ii*noDims + jj] = xmins[jj];
xmaxes[ii*noDims + jj] = xmaxes[jj];
}
}

free(xmin);
free(xmax);
}

for (int ii = 0; ii < noDims; ii++) {
xmins[noControls*noDims + ii] = xmins[ii];
xmaxes[noControls*noDims + ii] = xmaxes[ii];
}

for (int ii = 1; ii < noControls; ii++) {
for (int jj = 0; jj < noDims; jj++) {
float xtemp = xmins[ii*noDims + jj];
if (xmins[noControls*noDims + jj] > xtemp) {
xmins[noControls*noDims + jj] = xtemp;
}

xtemp = xmaxes[ii*noDims + jj];
if (xmaxes[noControls*noDims + jj] < xtemp) {
xmaxes[noControls*noDims + jj] = xtemp;
}
}
}
}