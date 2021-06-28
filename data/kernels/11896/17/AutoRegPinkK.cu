#include "includes.h"


// includes, project

#define PI 3.1415926536f


int MaxThreadsPerBlock;
int MaxThreadsX;
int MaxThreadsY;


// Conversion d'un vecteur réel en vecteur complexe

// Conversion d'un vecteur complexe en vecteur réel


// Multiplie point par point un vecteur complex par un vecteur réel

// Applique y = at*x +bt à chaque point d'un vecteur réel




// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de réel
// Alpha n'est pas modifié

// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de bytes
// Alpha n'est pas modifié


// Remplissage de la linearmem (tableau de pixels) associée à la texture avec le tableau de réel
// Alpha autorise l'affichage au dessus d'un certain seuil


// Processus auto-régressif X2 = a*X1 + b*X0 + N0;



// Expansion
// On applique une interpolation bi-linéaire à la source

// Transformation Cartesian To Polar
// On applique une interpolation bi-linéaire à la source



__global__ void AutoRegPinkK(double* X0, double* X1, double* Y, double* C0, double* D0, double* LastF, int numElts, int Nc)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElts)
{
for (int j=0; j<Nc; j++)
LastF[j*numElts+i] = (X0[i]+X1[i])*D0[j] - C0[j]*LastF[j*numElts+i];

double w = 0;
for (int j=0; j<Nc; j++) w = w+ LastF[j*numElts+i];
Y[i] = w;
}
}