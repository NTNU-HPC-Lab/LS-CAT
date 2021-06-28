#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void computePathStates(int noPaths, int noDims, int nYears, int noControls, int year, float unitCost, float unitRevenue, int* controls, int noFuels, float *fuelCosts, float *uResults, float *uComposition, int noUncertainties, int *fuelIdx, int noCommodities, float* aars, float* totalPops, float* xin, int* currControls) {

// Global thread index
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPaths) {

// 1. Adjusted population for each species
// We only take the highest flow's adjusted population as this is a
// measure of how damaging the road is. If we instead used the aar of
// the random control selected, we would get overlaps in the optimal
// control map.
for (int ii = 0; ii < noDims-1; ii++) {
//            xin[idx*noDims + ii] = totalPops[idx*(noDims-1)*(nYears+1) + year*
//                    (noDims-1) + ii]*aars[idx*(nYears+1)*noControls*(noDims-1)
//                    + year*noControls*(noDims-1) + ii*noControls + controls[
//                    idx*nYears + year]];
xin[idx*noDims + ii] = totalPops[idx*(noDims-1)*(nYears+1) + year*
(noDims-1) + ii]*aars[idx*(nYears+1)*noControls*(noDims-1)
+ year*noControls*(noDims-1) + ii*noControls + (noControls
- 1)];
}

// 2. Unit profit
float unitFuel = 0.0;
float orePrice = 0.0;

// Compute the unit fuel cost component
for (int ii = 0; ii < noFuels; ii++) {
unitFuel += fuelCosts[ii]*uResults[idx*(nYears+1)*noUncertainties +
(year)*noUncertainties + fuelIdx[ii]];
}
// Compute the unit revenue from ore
for (int ii = 0; ii < noCommodities; ii++) {
orePrice += uComposition[idx*nYears*noCommodities + (year)*
noCommodities + ii]*uResults[idx*(nYears+1)*noUncertainties +
(year)*noUncertainties + noFuels + ii];
}

xin[idx*noDims + noDims-1] = unitCost + unitFuel - unitRevenue*
orePrice;
currControls[idx] = controls[idx*nYears + year];

//        printf("%f %f\n",unitFuel,orePrice);
}
}