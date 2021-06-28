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
__global__ void Kernel_norme(float *Vd, float *Nd, int *Vp, int *Vnp, int N, int C, int nr_max_val)
{
long int x = threadIdx.x + blockIdx.x * blockDim.x;

int pos;

if(x < N)
{
float norma = 0;

int Nr_val = Vnp[x];

for(int i = 0; i < Nr_val; i++)
{
pos = Vp[x * nr_max_val + i];
norma = norma + (Vd[x * C + pos] * Vd[x * C + pos]);
}

Nd[x] = norma;
}

}