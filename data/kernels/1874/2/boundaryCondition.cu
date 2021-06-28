#include "includes.h"
__global__ void boundaryCondition(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3) {
d_u1[0] = d_u1[1];
d_u2[0] = -d_u2[1];
d_u3[0] = d_u3[1];
d_u1[nbrOfGrids - 1] = d_u1[nbrOfGrids - 2];
d_u2[nbrOfGrids - 1] = -d_u2[nbrOfGrids - 2];
d_u3[nbrOfGrids - 1] = d_u3[nbrOfGrids - 2];
}