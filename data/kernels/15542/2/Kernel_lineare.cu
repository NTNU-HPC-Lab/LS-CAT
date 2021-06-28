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
__global__ void Kernel_lineare(float *Vd, float *Ris, int N, int C, int dim_indici, int *ind, int *Vp, int *Vnp, int nr_max_val)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

int j;
int pos;
int tmp_ind;
float lin;

for ( ; x < N ; x+=blockDim.x * gridDim.x)
{
for( ; y < dim_indici; y+=blockDim.y * gridDim.y)
{
tmp_ind = ind[y];

lin = 0.0;

int Nr_val = Vnp[x];

for(j = 0; j < Nr_val; j++)
{
pos = Vp[x * nr_max_val + j];
lin =  lin + (Vd[x * C + pos] * Vd[tmp_ind * C + pos]);
}

//Ris[x * dim_indici + y] = lin;
Ris[y * N + x ] = lin;
}
}
}