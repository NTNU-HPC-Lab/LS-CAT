#include "includes.h"
// cuDEBYE SOURCE CODE VERSION 1.5
// TO DO:
// - REWRITE TO DOUBLE PRECISION DISTANCE CALCULATIONS FOR BENCHMARKING
// - CONSIDER NOT CALLING SQRT (HISTOGRAM OF VALUE UNDER SQUARE -> problem with memory, no solution jet) IN KERNEL TO SAVE COMPUTATION TIME
// - USE INTEGER VALUES INSTEAD OF FLOAT AND CALCULATE IN FEMTO METERS INSTEAD OF ANGSTROM -> INTEGER OPERATIONS SHOULD REPLACE ROUND AND SINGLE PRECISION OPERATIONS WITH ACCEPTABLE ERROR
// - IMPLEMENT A CLEVER ALGORYTHM TO SET GRID AND BLOCK SIZE AUTOMATICALLY
// - BINARY FILE SUPPORT FOR FASTER INFORMATION EXCHANGE AND LESS MEMORY CONSUMPTION OR/AND PYTHON7MATLAB INTERFACE TO GET ARRAYS DIRECTLY
// - CREATE INTERFACE TO DISCUS (READ DISCUS STRUCTURES)
// - IMPLEMENT USAGE OF MORE GPU'S
// - MULTIPLE EMPTY LINES IN ASCII CAN CAUSE A CRASH DURING READING
// - HOST AND THRUST OPERATIONS ARE VERY INEFFICIENT (BUT FAST ENOUGH) -> MAYBE REWRITE THEM
// - ELIMINATE COMPILER WARNINGS FOR A MORE STABLE PROGRAM


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PREAMBLE: LIBARIES AND USEFULL BASIC FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Include cuda libaries for parallel computing

// Thrust libaries from the cuda toolkit for optimized vector operations

// Libaries for input and output streams for display results and read and write files.
// Better than the old printf shit
using namespace std;	// Normally all stream functions have to called via prefix std:: -> So functions can called withaout prefix (Example: std::cout -> cout)

// Libary for measuring calculation time

// define the mathematical constant pi
# define PI 3.14159265358979323846

// Function to check if input file parsed via commandline exists
__global__ void atomicScatter(int type1, int type2, int size_K, double *occ, double *beq, double *K, double *a, double *b, double *c, double *ffoobb) {
// Kernel is executed for each K/TwoTheta (one dimensional grid)
int Idx = blockIdx.x*blockDim.x + threadIdx.x;
// Only execute if K/TwoTheta exists and is no phantom value, caused be discrete grid and block size.
if (Idx < size_K) {
double rp16pi2 = -0.006332573977646; // = (-1) * 1/(16*pi²)
double negativeHalfSquaredS = K[Idx] * K[Idx] * rp16pi2; // = -sin²(theta)/lambda², s = 2*sin(theta)/lambda = 1/d
// Calculate occupancy and debye-waller part of the prefactor
ffoobb[Idx] = occ[type1] * occ[type2];
ffoobb[Idx] = ffoobb[Idx] * exp(negativeHalfSquaredS*(beq[type1] + beq[type2]));
// Calculate atomic scattering factords from 11 parameter approximation.
double f1 = c[type1];
double f2 = c[type2];
for (int i = 0; i < 5; i++) {
f1 += a[type1 * 5 + i] * exp(b[type1 * 5 + i] * negativeHalfSquaredS);
f2 += a[type2 * 5 + i] * exp(b[type2 * 5 + i] * negativeHalfSquaredS);
}
// Complement prefactor with calculated scattering factors
ffoobb[Idx] = ffoobb[Idx] * f1*f2;
}
}