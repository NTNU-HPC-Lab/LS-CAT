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
__global__ void UpdateCostDrumuri(Nod *noduri, int *costuriTemporale, int *costuriFinale)
{
int nod = threadIdx.x;
if (costuriTemporale[nod] > costuriFinale[nod])
{
costuriTemporale[nod] = costuriFinale[nod];
noduri[nod].vizitat = false;
}
costuriFinale[nod] = costuriTemporale[nod];
}