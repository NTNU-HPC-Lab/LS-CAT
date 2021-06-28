#ifndef GRID_H_
#define GRID_H_

#include "helpers.cuh"
#include <cufft.h>

struct Grid{
    float *rho;
    float *Ex;
    float *Ey;
    float *Ez;

    float *d_rho;
    float *d_Ex;
    float *d_Ey;
    float *d_Ez;

    float *d_Rrho;
    float *d_REx;
    float *d_REy;
    float *d_REz;

    float *sum_results;
    float *d_sum_results;


    float dx;
    float dy;
    float dz;

    int N_grid;
    int N_grid_all;

    dim3 gridBlocks;
    dim3 gridThreads;

    //fourier transformed versions of grid quantities, for fields solver
    cufftComplex *d_F_rho;
    cufftComplex *d_F_Ex;
    cufftComplex *d_F_Ey;
    cufftComplex *d_F_Ez;

    //instructions for cuFFT
    cufftHandle plan_forward;
    cufftHandle plan_backward;

    //the wave vector, for the field solver
    float *kv;
    float *d_kv;
    float rho_total;
    float E_total;
};

__global__ void solve_poisson(float *d_kv, cufftComplex *d_F_rho, cufftComplex *d_F_Ex, cufftComplex *d_F_Ey, cufftComplex *d_F_Ez, int N_grid, int N_grid_all);
__global__ void real2complex(float *input, cufftComplex *output, int N_grid, int N_grid_all);
__global__ void complex2real(cufftComplex *input, float *output, int N_grid, int N_grid_all);
__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez, int N_grid, int N_grid_all);
__global__ void set_grid_array_to_value(float *arr, float value, int N_grid, int N_grid_all);

void init_grid(Grid *g, int N_grid);
void reset_rho(Grid *g);
void field_solver(Grid *g);
void dump_density_data(Grid *g, char* name);

void debug_field_solver_uniform(Grid *g);
void debug_field_solver_sine(Grid *g);
void grid_cleanup(Grid *g);

#endif
