#include "includes.h"
/**
* Programma che simula il comportamento del gpdt per
* la risoluzione di un kernel di una serie di
* valori di dimensione variabile utilizzando la
* tecnologia cuda.
* compilare con:
* nvcc -o simil_gpdt_si_cuda simil_gpdt_si_cuda.cu
* lanciare con:
* ./simil_gpdt_si_cuda [numero vettori] [numero componenti] [numero di righe da calcolare] [tipo di kernel] [grado(int)/sigma(float)]
**/

using namespace std;

/**
* Funzione che riempie i vettori con numeri
* casuali compresi tra 0 e 99.
**/
__global__ void Kernel_gaus(float *Vd, float *Ris, float *Nd, int N, int C, int dim_indici, int *ind, float sigma, int *Vp, int *Vnp, int nr_max_val)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

int j;
int pos;
int tmp_ind;
float gaus;

for ( ; x < N ; x+=blockDim.x * gridDim.x)
{
for( ; y < dim_indici; y+=blockDim.y * gridDim.y)
{
tmp_ind = ind[y];

gaus = 0.0;

int Nr_val = Vnp[x];

for(j = 0; j < Nr_val; j++)
{
pos = Vp[x * nr_max_val + j];
gaus = gaus + (Vd[x * C + pos] * Vd[tmp_ind * C + pos]);
}

gaus = - 2.0*gaus +Nd[x] + Nd[tmp_ind];
gaus = (exp(-gaus*sigma));

//Ris[x * dim_indici + y] = gaus;
Ris[y * N + x] = gaus;
}
}
}