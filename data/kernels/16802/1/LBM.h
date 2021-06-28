/*
*   The Lattice Boltzmann Method with ETHD convection
*   Yifei Guan
*   Rice University
*   Apr/12/2020
*
*/
#ifndef __LBM_H
#define __LBM_H
#include <math.h>
#include <cufft.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ double test;

__device__ int perturb = 1;

int iteractionCount = 0;
double *T = (double*)malloc(sizeof(double));
double *M = (double*)malloc(sizeof(double));
double *C = (double*)malloc(sizeof(double));
double *Fe = (double*)malloc(sizeof(double));
double *Pr = (double*)malloc(sizeof(double));


unsigned int flag = 0; // if flat == 1, read previous data, otherwise initialize
const int nThreads = 20; // can divide NX

// define grids
const unsigned int NX = 200; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int W =  NX;
const unsigned int NY = 8; // number of grid points in y-direction, meaning NY-1 cells
const unsigned int NZ = 101;
const unsigned int H =  NZ;
const unsigned int NE = 2 * (NZ - 1);
const unsigned int size = NX*NY*NE;
__constant__ double LL = 2.0;
__constant__ double Lx = 2.0;
__constant__ double Ly = 0.08;
__constant__ double Lz = 1.0;
__constant__ double dx = 1.0 / 100.0; //need to change according to NX and LX
__constant__ double dy = 1.0 / 100.0; //need to change according to NY and LY
__constant__ double dz = 1.0 / 100.0; //need to change according to NZ and LZ
// define physics
double uw_host = 0.0; // velocity of the wall
double exf_host = 0.0; // external force for poisseuille flow
__device__ double uw;
__device__ double exf;
__constant__ double CFL = 0.01; // CFL = dt/dx
__constant__ double dt = 0.01*1.0 / 100.0; // dt = dx * CFL need to change according to dx, dy
__constant__ double cs_square = 1.0 / 3.0 / (0.01*0.01); // 1/3/(CFL^2)
__constant__ double rho0 = 1600.0;
__constant__ double charge0 = 10.0;
__constant__ double voltage = 1.0e4;
__constant__ double eps = 1.0e-4;
__constant__ double K = 2.5e-5;
__constant__ double diffu = 2.5e-4;
double nu_host = 0.25;
__device__ double nu = 0.25;

// Thermal properties

__device__ double D = 0.025; // Thermal diffusivity
double Ra_host = 6000;
__device__ double Ra = 6000; // Rayleigh number
__device__ double TH = 1; // Temperature of the lower surface


// define scheme
const unsigned int ndir = 27;
const size_t mem_size_0dir = sizeof(double)*NX*NY*NZ;
const size_t mem_size_n0dir = sizeof(double)*NX*NY*NZ*(ndir - 1);
const size_t mem_size_scalar = sizeof(double)*NX*NY*NZ;
const size_t mem_size_ext_scalar = sizeof(double)*NX*NY*NE;

// weights of populations (total 9 for D2Q9 scheme)
__constant__ double w0  = 8.0 / 27.0;  // zero weight for i=0
__constant__ double ws  = 2.0 / 27.0;  // adjacent weight for i=1-6
__constant__ double wa  = 1.0 / 54.0;  // adjacent weight for i=7-18
__constant__ double wd  = 1.0 / 216.0; // diagonal weight for i=19-26

// parameters for (two-relaxation time) TRT scheme
__constant__ double V  = 1.0 / 12.0;
__constant__ double VC = 1.0e-6;
__constant__ double VT = 1.0 / 12.0;


const unsigned int NSTEPS = 1500000;
const unsigned int NSAVE =  NSTEPS / 10;
const unsigned int NMSG   =  NSAVE;
const unsigned int printCurrent = 5000;


// physical time
double t;

double *f0_gpu, *f1_gpu, *f2_gpu;
double *h0_gpu, *h1_gpu, *h2_gpu;
double *temp0_gpu, *temp1_gpu, *temp2_gpu;
double *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
double *charge_gpu, *phi_gpu, *T_gpu;
double *Ex_gpu, *Ey_gpu, *Ez_gpu;
double *kx, *ky, *kz;
cufftHandle plan = 0;
cufftDoubleComplex *freq_gpu_ext, *charge_gpu_ext, *phi_gpu_ext;
double *f0bc; // store f0 of the lower plate for further use
double *kx_host = (double*)malloc(sizeof(double)*NX);
double *ky_host = (double*)malloc(sizeof(double)*NY);
double *kz_host = (double*)malloc(sizeof(double)*NE);
double dt_host;
double Lx_host;
double Ly_host;
double dy_host;
double Lz_host;
double dz_host;
double *charge_host = (double*)malloc(mem_size_scalar);
double *Ez_host = (double*)malloc(mem_size_scalar);

// suppress verbose output
const bool quiet = true;

void initialization(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void read_data(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

void init_equilibrium(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);

void stream_collide_save(double*,double*,double*,double*,double*,double*, double*, double*, double*, 
	double*, double*, double*, double*, double*, double*, double*, double*,double*,double,double*);
void report_flow_properties(unsigned int,double,double*,double*,double*,double*, double*, double*, double*, double*, double*);
void save_scalar(const char*,double*,double*,unsigned int);
void save_data_tecplot(FILE*, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,int);
void save_data_end(FILE*, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void compute_parameters(double*, double*, double*, double*, double*);
void extension(double*, cufftDoubleComplex*);
void efield(double*, double*, double*, double*);
void derivative(double*, double*, double*, cufftDoubleComplex*);
void extract(double*, cufftDoubleComplex*);

double current(double*, double*);
void record_umax(FILE*, double, double*, double*, double*);

inline size_t scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
	return NX*(NY*z+y) + x;
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}
#endif /* __LBM_H */

