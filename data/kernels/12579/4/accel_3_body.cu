#include "includes.h"
/*
*  This program is a CUDA C program simulating the N-body system
*    of two galaxies as PHY 241 FINAL PROJECTS
*
*/

/*
*  TODO:
*    1. andromeda
*    2. For accel of center of A, only consider accel from center of B. The same for B.
*    3. When the distance between A and B, the soft parameter changed to 0.2Rmin
*    4. report
*    5. presentation
*
*/



/*
**  Modify the constant parameters if neccessary
**    Constant Section
*/
#define PI 3.14159265
#define BUFFERSIZE 256
#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif
//#define SOFTPARAMETER 0.2 * RMIN
// #define AU 149597870700.0
// #define R (77871.0 * 1000.0 / AU)
// #define G (4.0 * pow(PI, 2))
#define G 0.287915013
#define MASS_1 1000              // Center mass of 1st galaxy
#define MASS_2 1000                // Center mass of 2nd galaxy
#define NUM_OF_RING_1 12         // Number of rings in 1st galaxy
#define NUM_OF_RING_2 12          // Number of rings in 2nd galaxy
// #define RING_BASE_1 (R * 0.2)       // Radius of first ring in 1st galaxy
// #define RING_BASE_2 (R * 0.2)       // Radius of first ring in 2nd galaxy
#define NUM_P_BASE 12             // Number of particles in the first ring
#define INC_NUM_P 3               // increment of number of particles each step
// #define INC_R_RING (0.5 * R)      // increment of radius of rings each step
#define PMASS 1             // mass of each particle
#define V_PARAMTER 1            // Parameter adding to initial velocity to make it elliptic
#define RMIN 1
#define ECCEN 0.5
#define RMAX ((1.0 + ECCEN) * RMIN / (1.0 - ECCEN))
#define RING_BASE_1 (RMIN * 0.2)       // Radius of first ring in 1st galaxy
#define RING_BASE_2 (RMIN * 0.2)       // Radius of first ring in 2nd galaxy
#define INC_R_RING (RMIN * 0.05)      // increment of radius of rings each step
#define SOFTPARAMETER 0.000001
/*
*  Major Function Declarations Section
*
*/


/*
*  Functions Implmenetation Section
*
*/

__global__ void accel_3_body(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass, double dt){
/*
*  Three body leapfrog: each particle is in a 3 body system with center mass of galaxy 1 and center mass of galaxy 2
*    Because of SOFTPARAMETER, we dont need to determine if thread is computing against itself
*/
const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numofp1 = NUM_P_BASE * NUM_OF_RING_1 + (NUM_OF_RING_1 - 1) * NUM_OF_RING_1 * INC_NUM_P / 2 + 1;
double ax = 0.0, ay = 0.0, az = 0.0, norm1, norm2;
#ifdef SOFTPARA
double tempsp = (pow(pow(x[0] - x[numofp1], 2) + pow(y[0] - y[numofp1], 2) + pow(z[0] - z[numofp1], 2), 1.5) <= 0.25 * RMIN) ? 0.5 * RMIN : SOFTPARAMETER;
double softparameter = (serial == 0 && serial == numofp1) ? tempsp : SOFTPARAMETER;
#else
double softparameter = 0.0;
#endif
norm1 = pow(softparameter + pow(x[serial] - x[0], 2) + pow(y[serial] - y[0], 2) + pow(z[serial] - z[0], 2), 1.5);
norm2 = pow(softparameter + pow(x[serial] - x[numofp1], 2) + pow(y[serial] - y[numofp1], 2) + pow(z[serial] - z[numofp1], 2), 1.5);
if(serial != 0){
ax += -G * mass[0] * (x[serial] - x[0]) / norm1;
ay += -G * mass[0] * (y[serial] - y[0]) / norm1;
az += -G * mass[0] * (z[serial] - z[0]) / norm1;
}
if(serial != numofp1){
ax += -G * mass[numofp1] * (x[serial] - x[numofp1]) / norm2;
ay += -G * mass[numofp1] * (y[serial] - y[numofp1]) / norm2;
az += -G * mass[numofp1] * (z[serial] - z[numofp1]) / norm2;
}
if(serial < n){
vx[serial] += 0.5 * dt * ax;
vy[serial] += 0.5 * dt * ay;
vz[serial] += 0.5 * dt * az;
}
}