#include "includes.h"
__global__ void accel_3_body(int n, double* x, double* y, double* z, double* vx, double* vy, double* vz, double* mass, double dt){
/*
*  Three body leapfrog: each particle is in a 3 body system with center mass of galaxy 1 and center mass of galaxy 2
*    Because of SOFTPARAMETER, we dont need to determine if thread is computing against itself
*/
const unsigned int serial = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numofp1 = NUM_P_BASE * NUM_OF_RING_1 + (NUM_OF_RING_1 - 1) * NUM_OF_RING_1 * INC_NUM_P / 2 + 1;
double ax = 0.0, ay = 0.0, az = 0.0, norm1, norm2;
#ifdef SOFTPARA
double tempsp = (pow(pow(x[0] - x[numofp1], 2) + pow(y[0] - y[numofp1], 2) + pow(z[0] - z[numofp1], 2), 1.5) <= RMIN) ? 0.2 * RMIN : SOFTPARAMETER;
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