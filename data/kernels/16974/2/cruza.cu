#include "includes.h"

//Bibliotecas Basicas

//Biblioteca Thrust


//Biblioteca cuRAND


//PARAMETROS GLOBAIS
const int QUANT_PAIS_AVALIA = 4;
int POP_TAM = 200;
int N_CIDADES = 20;
int BLOCKSIZE = 1024;
int TOTALTHREADS = 2048;
int N_GERA = 100;
const int MUT = 10;
const int MAX = 19;
const int MIN = 0;
const int ELITE = 2;

/*
* Busca por erros nos processos da gpu
*/
__global__ void cruza(unsigned int n, unsigned int np, int *cidadesAle, int *pop, int *newPop, int *poolPais, int *mutacoes) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

int paiA, paiB, copiaPai, crossover, mutar, pontoMutar;

for (int i=index; i<n; i+=stride) {
copiaPai = cidadesAle[i*4];
crossover = cidadesAle[(i+1)*4] % np;
mutar = cidadesAle[(i+2)*4];
pontoMutar = cidadesAle[(i+3)*4] % np;
paiA = poolPais[i];
paiB = poolPais[i+1];

if (copiaPai < ELITE) {
for (int j=0; j<np; j++) {
newPop[(i*np) + j] = pop[(paiA*np) + j];
continue;
}
}
for(int j=0;j<np;j++)
{
newPop[(i*np) + j] = pop[(paiA*np) + j];
}
int t=0, aux=0, crossoverSup;
crossoverSup=(crossover +mutacoes[i]>MAX)?(MAX):(crossover +mutacoes[i]);
for(int j=crossover; j<crossoverSup;j++)
{
t=0;
while(newPop[(i*np) +t]!=pop[(paiB*np) + j])
{
t++;
}
aux = newPop[i*np+j];
newPop[i*np+j] = newPop[i*np+t];
newPop[i*np+t] = aux;

}

if (mutar < MUT) {
int mut = (mutacoes[i]>MAX)?(MAX):((mutacoes[i]<MIN)?(MIN):(mutacoes[i]));
t=0;
while(newPop[(i*np) +t]!=mut)
{
t++;
}
aux = newPop[i*np+pontoMutar];
newPop[i*np+pontoMutar] = newPop[i*np+t];
newPop[i*np+t] = aux;

}

}

}