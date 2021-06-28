#include "includes.h"
/*
* Rayhana ZIARA
* produit matrice vecteur
*/


/*
* DESCRIPTION : kernel concernant le produit matrice vecteur
* PARAMETRES : matrice A, vecteur v, vecteur r et taille des vecteurs
* RETOUR : /
*/

/*
* DESCRIPTION : fonction d'affichage de matrice et de vecteur
* PARAMETRES : matrice à afficher, nb ligne et nb colonne de A,
* RETOUR : /
*/
__global__ void matVect(float *A, float *v, float *r, int size)
{
float resultat = 0.0;
int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index > size)
{
printf("ERREUR - Index > size\n");
return;
}

for(int i = 0; i < size; i++)
resultat += A[i * size + index] * v[i];

r[index] = resultat;
}