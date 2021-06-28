#include "includes.h"
__global__ void mandelKernel(double planoFactorXd, double planoFactorYd, double planoVxd, double planoVyd, int maxIteracionesd, unsigned int *coloresd, int img_width, int img_height, int num_processes, int my_pid, int rw) {
int columna, fila;
double X, Y;
double pReal = 0.0;
double pImag = 0.0;
double pRealAnt, pImagAnt, distancia;

// Determine pixel
columna = blockIdx.x * blockDim.x + threadIdx.x;
fila = (rw * MAX_ROWS_PER_KERNEL) + (blockIdx.y * blockDim.y) + threadIdx.y;
int real_row = (fila * num_processes) + my_pid;

if(real_row >= img_height)
return;

// Real pixel coords
X = (planoFactorXd * (double)columna) + planoVxd;
Y = (planoFactorYd * ((double)(img_height - 1) - (double)real_row)) + planoVyd;
int i = 0;
do {
pRealAnt = pReal;
pImagAnt = pImag;
pReal = ((pRealAnt * pRealAnt) - (pImagAnt * pImagAnt)) + X;
pImag = (2.0 * (pRealAnt * pImagAnt)) + Y;
i++;
distancia = pReal*pReal + pImag*pImag;
}while ((i < maxIteracionesd) && (distancia <= 4.0));
if(i == maxIteracionesd) i = 0;
coloresd[(fila * img_width) + columna] = i;
}