#include "includes.h"


#define NX 100                          // No. of cells in x direction
#define NY 100                          // No. of cells in y direction
#define NZ 100                          // No. of cells in z direction
#define N (NX*NY*NZ)            // N = total number of cells in domain
#define L 100                             // L = length of domain (m)
#define H 100                             // H = Height of domain (m)
#define W 100                             // W = Width of domain (m)
#define DX (L/NX)                       // DX, DY, DZ = grid spacing in x,y,z.
#define DY (H/NY)
#define DZ (W/NZ)
#define DT 0.001                       // Time step (seconds)

#define R (1.0)           // Dimensionless specific gas constant
#define GAMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

//#define DEBUG_VALUE

float *dens;              //density
float *temperature;        //temperature
float *xv;                //velocity in x
float *yv;                //velocity in y
float *zv;                //velocity in z
float *press;             //pressure

float *d_dens;              //density
float *d_temperature;       //temperature
float *d_xv;                //velocity in x
float *d_yv;                //velocity in y
float *d_zv;                //velocity in z
float *d_press;             //pressure

float *U;
float *U_new;
float *E;
float *F;
float *G;
float *FF;
float *FB;
float *FR;
float *FL;
float *FU;
float *FD;

float *h_body;
float *d_body;

int total_cells = 0;            // A counter for computed cells


__global__ void GPUHeatContactFunction(float *a, float *b, int *body) {

}