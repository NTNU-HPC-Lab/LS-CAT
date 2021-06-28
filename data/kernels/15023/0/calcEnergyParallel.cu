#include "includes.h"
/*
This script is a mockup of the fuctionality to be parallelized in the Monte Carlo
Simulation. It calculates "energy" among pairs of "atoms" in a system, and compares
serial and parallel performance.

The command line arguments are as follows:
first argument (optional) - integer representing number of atoms
- defaults to 100
- input -1 to run benchmarking suite for
10000 <= N <= 40000 and specified thread
block size
second argument (optional) - integer <= 1024 representing thread block size
- input -1 to run benchmarking suite for
64 <= BS <= 1024 and specified N value
For example, -1 512 will run all N with block size = 512
-1 or -1 -1 will run all N for all block sizes
20000 -1 will run N = 20000 for all block sizes

Each simulation adds a line into RunLog.log with data about the run.
*/

//Given two indices in an array (representing atoms),
//calculate their product (potential energy),
//and store in energies array.
//Parallel

//Given two indices in an array (representing atoms),
//calculate their product (potential energy),
//and store in energies array.
//Serial
__global__ void calcEnergyParallel(int *atoms, int numAtoms, int *energies, int numEnergies)
{
int atom1 = blockIdx.x, atom2 = blockIdx.y * blockDim.x + threadIdx.x,
energyIdx;

if (atom2 < numAtoms && atom2 > atom1)
{
energyIdx = gridDim.x * atom1 + atom2 - (blockIdx.x + 1) * (blockIdx.x + 2) / 2;
energies[energyIdx] = atoms[atom1] * atoms[atom2];
}
}