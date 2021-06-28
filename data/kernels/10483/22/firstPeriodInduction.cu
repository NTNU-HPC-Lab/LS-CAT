#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void firstPeriodInduction(int noPaths, int nYears, int noSpecies, int noControls, float timeStep, float unitCost, float unitRevenue, float rrr, int noFuels, int noCommodities, float* Q, float* fuelCosts, float* totalPops, float* speciesParams, int* controls, float* aars, float* uComposition, float* uResults, int* fuelIdx, float* condExp, int* optCont, float* stats) {

float *payoffs, *dataPoints;
payoffs = (float*)malloc(noControls*sizeof(float));
dataPoints = (float*)malloc(noControls*sizeof(float));
bool* valid;
valid = (bool*)malloc(noControls*sizeof(bool));

float unitFuel = 0.0;
float orePrice = 0.0;

// Compute the unit fuel cost component
for (int ii = 0; ii < noFuels; ii++) {
unitFuel += fuelCosts[ii]*uResults[fuelIdx[ii]];
}
// Compute the unit revenue from ore
for (int ii = 0; ii < noCommodities; ii++) {
orePrice += uComposition[ii]*uResults[noFuels + ii];
}

for (int ii = 0; ii < noControls; ii++) {
dataPoints[ii] = 0.0;
payoffs[ii] = 0.0;
}

// Now get the average payoff across all paths of the same control for
// each control

for (int ii = 0; ii < noPaths; ii++) {
int control = controls[ii*nYears];

payoffs[control] += condExp[ii+noPaths];
dataPoints[control]++;
}

for (int ii = 0; ii < noControls; ii++) {
// Compute the single period financial payoff for each control
// for this period and the adjusted profit. If any adjusted
// population is below the threshold, then the payoff is
// invalid.
if (dataPoints[ii] > 0) {
payoffs[ii] = payoffs[ii]/(dataPoints[ii]*(1+rrr*timeStep/
100));
} else {
break;
}

valid[ii] = true;
for (int jj = 0; jj < noSpecies; jj++) {
float adjPop = totalPops[jj]*aars[jj*noControls + ii];

// Zero flow control is always valid
if (adjPop < speciesParams[noSpecies*jj + 3] && ii > 0) {
valid[ii] = false;
break;
}
}

// Compute the payoff for the control if valid.
if (valid[ii]) {
// Now compute the overall period profit for this control
// given the prevailing stochastic factors (undiscounted).
payoffs[ii] += Q[ii]*(unitCost + unitFuel - unitRevenue*
orePrice);

// Take care of regression anomalies
if (payoffs[ii] > 0) {
payoffs[ii] = 0.0;
}
} else {
payoffs[ii] = NAN;
}
}

//    printf("Pop: %6.2f %6.2f %6.2f\n", totalPops[0]*aars[0], totalPops[0]*aars[1],totalPops[0]*aars[2]);

// The optimal value is the one with the lowest net present cost.
// As the zero flow rate option is always available, we can
// initially set the optimal control to this before checking the
// other controls.
float bestExp = payoffs[0];
int bestCont = 0;

for (int ii = 1; ii < noControls; ii++) {
if (isfinite(payoffs[ii])) {
if (payoffs[ii] < bestExp) {
bestExp = payoffs[ii];
bestCont = ii;
}
}
}

// Assign the optimal control and payoff to all paths at time period 0

// Standard deviation
stats[2] = 0;

// Assign values and prepare standard deviation
for (int ii = 0; ii < noPaths; ii++) {
condExp[ii] = bestExp;
optCont[ii] = bestCont;

if (controls[ii*nYears] == bestCont) {
stats[2] += (condExp[ii+noPaths] - payoffs[bestCont])*(condExp[ii
+noPaths] - payoffs[bestCont]);
}
}

stats[0] = condExp[0];
stats[1] = (float)optCont[0];
stats[2] = sqrt(stats[2]/(dataPoints[bestCont]*(1+rrr/(100*timeStep))));

free(valid);
free(payoffs);
free(dataPoints);
}