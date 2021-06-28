#include "includes.h"

#define iceil(num, den) (num + den - 1) / den
#define ARRAY_SIZE 20 //must be an even number; this number/2 = number of points //sets random array and constant mem size
//#define BIN 100 //divides the grid into square bins to vote on. perfect square value
#define NUM_LINES 4 //top X voted lines. Picks first X Largest from top left to bottom right of grid space.

/*GRID evaluated for bin voting
* Must always be a square grid with origin at center
*/
#define dimension 5
#define LXBOUND (-1*dimension) //lowest X
#define RXBOUND (dimension) //highest X
#define LYBOUND (-1*dimension) //lowest Y
#define UYBOUND (dimension) //highest Y
////////////////////////////////

#define INCREMENT 1 //precision, length of 1 side of the square(bin)
//The (abs)difference between between two sides is the length of the grid. Length/Increment determines how many bins

#define column (((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) / ((RXBOUND + UYBOUND) / INCREMENT)

__constant__ int d_coordarray[ARRAY_SIZE];//Place coordinates in constant memory

//show grid with votes. Becomes unuseful when bins > 20x20
__global__ void kernelHough(int size, int* d_binarray) {
/*
take a piece of the array. discretize into y=mx+b format per point. check all points and increment all bins touched
at the end recombine all shared memory to a global bin tally. Take the most significant X numbers as lines.
discretized from point(1,1) ==(m,n)==> (-1,1)
check each bin for count and sum them to a global array in sync
NUM of coordinates will check all bins for their own equation and increment appropriately
*/

// Number from 0 through arraysize / 2
const int thread = 2 * (blockDim.x * blockIdx.x + threadIdx.x);

// Slope is discretized space = -x
const float slope = -1.0 * d_coordarray[thread];

// Intercept in discretized space = y
const float intercept = d_coordarray[thread + 1];

int counter = 0;//keeps current array index being checked
//loop through entire graph
for (float x = LXBOUND; x < RXBOUND; x += INCREMENT) {

const float xMin = x;
const float xMax = x + INCREMENT;

for (float y = UYBOUND; y > LYBOUND; y -= INCREMENT) {

const float yMin = y - INCREMENT;
const float yMax = y;
//calculates possible y range associated with the known x range
const float lower_range = slope * xMin + intercept;
const float upper_range = slope * xMax + intercept;
//if the possible y ranges corresponding to the x values exist within the actual y range increment bin
if ((lower_range <= yMax && lower_range >= yMin) || (upper_range <= yMax && upper_range >= yMin))
atomicAdd(&d_binarray[counter], 1);//increment bin, protected from race condition

counter++;
}
}
}