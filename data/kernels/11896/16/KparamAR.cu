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



__global__ void  KparamAR(double* a, double* b, double* c, double ss, double dtAR, int width, int height)
{
#define eps 1E-12;

int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if (i >= width || j >= height) return;

int  x= i;
int  y= j;

if (i> width/2)  x = width-i;
if (j> height/2) y = height-j;

double r = sqrt( (double)x*x + (double)y*y )+Eps;

a[i+j*width] =  2-dtAR*2*ss*r- pow(dtAR*ss*r,2);
b[i+j*width] = -1+dtAR*2*ss*r;
// c[i+j*width] =  50* pow(dtAR,2);
// Correction Jonathan 7-12-16
c[i+j*width] =  1;

}