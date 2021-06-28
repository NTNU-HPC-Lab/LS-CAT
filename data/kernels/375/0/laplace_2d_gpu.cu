#include "includes.h"
/* Programmaufruf mit 2 Argumenten:
1. Größe des Gitters (mit Rand): Nx+2 (= Ny+2)
2. Dimension eines Cuda-Blocks: dim_block (findet nur Anwendung, wenn Nx+2 > dim_block)
*/

/*
Globale Variablen stehen in allen Funktionen zur Verfuegung.
Achtung: Das gilt *nicht* fuer Kernel-Funktionen!
*/
int Nx, Ny, npts;
int *active;

/*
Fuer die Koordinaten:
i = 0,1,...,Nx+1
j = 0,1,...,Ny+1
wird der fortlaufenden Index berechnet
*/
__global__ void laplace_2d_gpu(double *w, double *v, const int nx, const int ny)
{
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
if (ix>0 && ix<(nx+1) && iy>0 && iy<(ny+1)) // Bedingung, dass nur innere Punkte berechnet werden
{
unsigned int idx = iy*(blockDim.x * gridDim.x) + ix;
w[idx] = 4*v[idx] - (v[idx-1] + v[idx+1] + v[(idx-(gridDim.x*blockDim.x))] + v[(idx+(gridDim.x*blockDim.x))]);
}
}