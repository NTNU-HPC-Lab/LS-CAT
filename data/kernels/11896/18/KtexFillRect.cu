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



__global__ void KtexFillRect(void* surface, double* tb, int width, int height, size_t pitch, float2* Pts, int k, float th)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

unsigned char *pixel1;

if (x >= width || y >= height) return;

pixel1 = (unsigned char *)( (char*)surface + y*pitch) + 4*x;

if (
((Pts[1].y-Pts[0].y)*(x-Pts[0].x)-( y-Pts[0].y)*(Pts[1].x-Pts[0].x)>=0)
&&
((Pts[2].y-Pts[1].y)*(x-Pts[1].x)-( y-Pts[1].y)*(Pts[2].x-Pts[1].x)>=0)
&&
((Pts[3].y-Pts[2].y)*(x-Pts[2].x)-( y-Pts[2].y)*(Pts[3].x-Pts[2].x)>=0)
&&
((Pts[0].y-Pts[3].y)*(x-Pts[3].x)-( y-Pts[3].y)*(Pts[0].x-Pts[3].x)>=0)
&&
(pixel1[k]>=th)
)
tb[x + width*y] = 1;



}