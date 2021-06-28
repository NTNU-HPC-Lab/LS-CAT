#include "includes.h"



#define NUMAR_NODURI 500

#define NUMAR_MUCHII 500

#define COST_MAXIM 1000000

typedef struct
{
int nod1;
int nod2;
} Muchie;

typedef struct
{
int nodId;
bool vizitat;
} Nod;

//Gaseste costul drumului de la nodul start la nodul stop
__device__ __host__ int CautareMuchie(Nod start, Nod stop, Muchie *muchii, int *costuri)
{
for (int i = 0; i < NUMAR_MUCHII; i++)
if (muchii[i].nod1 == start.nodId && muchii[i].nod2 == stop.nodId)
return costuri[i];

return COST_MAXIM;
}
__global__ void Cauta_Nod(Nod *noduri, Muchie *muchii, int *costuri, int *costTemporal, int *costFinal)
{
int nod = threadIdx.x;
if (noduri[nod].vizitat == false)
{
noduri[nod].vizitat = true;
for (int n = 0; n < NUMAR_NODURI; n++)
{
//Cauta costul muchiei
int cost = CautareMuchie(noduri[nod], noduri[n], muchii, costuri);

//ia costul minim
if (costFinal[n] > costTemporal[nod] + cost && cost < COST_MAXIM)
costFinal[n] = costTemporal[nod] + cost;
}
}
}