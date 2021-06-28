#include "includes.h"
__global__ void normalize_energy_gpu(float *ksn2e, float *ksn2f, double omega_re, double omega_im, float *nm2v_re, float *nm2v_im, int nfermi, int norbs, int nvirt, int vstart)
{
int i = blockIdx.x * blockDim.x + threadIdx.x; //nocc
int j = blockIdx.y * blockDim.y + threadIdx.y; //nvirt
float en=0.0, fn=0.0, em=0.0, fm=0.0, old_re, old_im;
double d1p, d1pp, d2p, d2pp, alpha, beta;

if (i < nfermi)
{
en = ksn2e[i];
fn = ksn2f[i];
if ( j < norbs - vstart )
{
em = ksn2e[j + vstart];
fm = ksn2f[j + vstart];

d1p = omega_re - (em-en); d1pp = omega_im;
d2p = omega_re + (em-en); d2pp = omega_im;

alpha = d1p/(d1p*d1p + d1pp*d1pp) - d2p/(d2p*d2p + d2pp*d2pp);
beta = -d1pp/(d1p*d1p + d1pp*d1pp) + d2pp/(d2p*d2p + d2pp*d2pp);
old_re = nm2v_re[i*nvirt + j];
old_im = nm2v_im[i*nvirt + j];

nm2v_re[i*nvirt + j] = (fn - fm)*(old_re*alpha - old_im*beta);
nm2v_im[i*nvirt + j] = (fn - fm)*(old_re*beta + old_im*alpha);
//printf("i = %d, j = %d, m = %d, alpha = %f, beta = %f, old_re = %f, old_im = %f, nm2v_re = %f, nm2v_im = %f\n",
//    i, j, m, alpha, beta, old_re, old_im, nm2v_re[index], nm2v_im[index]);

//nm2v = nm2v * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /(comega + (em - en)) );
}
}
}