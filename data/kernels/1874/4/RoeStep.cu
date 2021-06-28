#include "includes.h"
__device__ void d_boundaryCondition(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3) {
d_u1[0] = d_u1[1];
d_u2[0] = -d_u2[1];
d_u3[0] = d_u3[1];
d_u1[nbrOfGrids - 1] = d_u1[nbrOfGrids - 2];
d_u2[nbrOfGrids - 1] = -d_u2[nbrOfGrids - 2];
d_u3[nbrOfGrids - 1] = d_u3[nbrOfGrids - 2];
}
__global__	void RoeStep(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3, const double *d_vol, double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama, double *w1,double *w2,double *w3,double *w4, double *fc1,double *fc2,double *fc3, double *fr1,double *fr2,double *fr3, double *fl1,double *fl2,double *fl3, double *fludif1,double *fludif2,double *fludif3, double *rsumr, double *utilde, double *htilde, double *uvdif, double *absvt, double *ssc, double *vsc, double *eiglam1,double *eiglam2,double *eiglam3, double *sgn1,double *sgn2,double *sgn3, int *isb1,int *isb2,int *isb3, double *a1,double *a2,double *a3, double *ac11,double *ac12,double *ac13, double *ac21,double *ac22,double *ac23) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < nbrOfGrids; i += stride) {

// find parameter vector w
{
w1[i] = sqrt(d_vol[i] * d_u1[i]);
w2[i] = w1[i] * d_u2[i] / d_u1[i];
w4[i] = (*d_gama - 1) * (d_u3[i] - 0.5 * d_u2[i] * d_u2[i] / d_u1[i]);
w3[i] = w1[i] * (d_u3[i] + w4[i]) / d_u1[i];
}

// calculate the fluxes at the cell center
{
fc1[i] = w1[i] * w2[i];
fc2[i] = w2[i] * w2[i] + d_vol[i] * w4[i];
fc3[i] = w2[i] * w3[i];
}

__syncthreads(); // because of the [i - 1] index below
// calculate the fluxes at the cell walls
if (i > 0) {
fl1[i] = fc1[i - 1]; fr1[i] = fc1[i];
fl2[i] = fc2[i - 1]; fr2[i] = fc2[i];
fl3[i] = fc3[i - 1]; fr3[i] = fc3[i];
}

// calculate the flux differences at the cell walls
if (i > 0) {
fludif1[i] = fr1[i] - fl1[i];
fludif2[i] = fr2[i] - fl2[i];
fludif3[i] = fr3[i] - fl3[i];
}

__syncthreads(); // because of the [i - 1] index below
// calculate the tilded state variables = mean values at the interfaces
if (i > 0) {
rsumr[i] = 1 / (w1[i - 1] + w1[i]);

utilde[i] = (w2[i - 1] + w2[i]) * rsumr[i];
htilde[i] = (w3[i - 1] + w3[i]) * rsumr[i];

absvt[i] = 0.5 * utilde[i] * utilde[i];
uvdif[i] = utilde[i] * fludif2[i];

ssc[i] = (*d_gama - 1) * (htilde[i] - absvt[i]);
if (ssc[i] > 0.0)
vsc[i] = sqrt(ssc[i]);
else {
vsc[i] = sqrt(abs(ssc[i]));
}
}

// calculate the eigenvalues and projection coefficients for each eigenvector
if (i > 0) {
eiglam1[i] = utilde[i] - vsc[i];
eiglam2[i] = utilde[i];
eiglam3[i] = utilde[i] + vsc[i];
sgn1[i] = eiglam1[i] < 0.0 ? -1 : 1;
sgn2[i] = eiglam2[i] < 0.0 ? -1 : 1;
sgn3[i] = eiglam3[i] < 0.0 ? -1 : 1;
a1[i] = 0.5 * ((*d_gama - 1) * (absvt[i] * fludif1[i] + fludif3[i]
- uvdif[i]) - vsc[i] * (fludif2[i] - utilde[i]
* fludif1[i])) / ssc[i];
a2[i] = (*d_gama - 1) * ((htilde[i] - 2 * absvt[i]) * fludif1[i]
+ uvdif[i] - fludif3[i]) / ssc[i];
a3[i] = 0.5 * ((*d_gama - 1) * (absvt[i] * fludif1[i] + fludif3[i]
- uvdif[i]) + vsc[i] * (fludif2[i] - utilde[i]
* fludif1[i])) / ssc[i];
}

// divide the projection coefficients by the wave speeds to evade expansion correction
if (i > 0) {
a1[i] /= eiglam1[i] + tiny;
a2[i] /= eiglam2[i] + tiny;
a3[i] /= eiglam3[i] + tiny;
}

// calculate the first order projection coefficients ac1
if (i > 0) {
ac11[i] = -sgn1[i] * a1[i] * eiglam1[i];
ac12[i] = -sgn2[i] * a2[i] * eiglam2[i];
ac13[i] = -sgn3[i] * a3[i] * eiglam3[i];
}

// apply the 'superbee' flux correction to made 2nd order projection coefficients ac2
{
ac21[1] = ac11[1];
ac21[nbrOfGrids - 1] = ac11[nbrOfGrids - 1];
ac22[1] = ac12[1];
ac22[nbrOfGrids - 1] = ac12[nbrOfGrids - 1];
ac23[1] = ac13[1];
ac23[nbrOfGrids - 1] = ac13[nbrOfGrids - 1];


double dtdx = *d_tau / *d_h;
if ((i > 1) && (i < nbrOfGrids - 1)) {
isb1[i] = i - int(sgn1[i]);
ac21[i] = ac11[i] + eiglam1[i] *
((fmax(0.0, fmin(sbpar1 * a1[isb1[i]], fmax(a1[i], fmin(a1[isb1[i]], sbpar2 * a1[i])))) +
fmin(0.0, fmax(sbpar1 * a1[isb1[i]], fmin(a1[i], fmax(a1[isb1[i]], sbpar2 * a1[i]))))) *
(sgn1[i] - dtdx * eiglam1[i]));
isb2[i] = i - int(sgn2[i]);
ac22[i] = ac12[i] + eiglam2[i] *
((fmax(0.0, fmin(sbpar1 * a2[isb2[i]], fmax(a2[i], fmin(a2[isb2[i]], sbpar2 * a2[i])))) +
fmin(0.0, fmax(sbpar1 * a2[isb2[i]], fmin(a2[i], fmax(a2[isb2[i]], sbpar2 * a2[i]))))) *
(sgn2[i] - dtdx * eiglam2[i]));
isb3[i] = i - int(sgn3[i]);
ac23[i] = ac13[i] + eiglam3[i] *
((fmax(0.0, fmin(sbpar1 * a3[isb3[i]], fmax(a3[i], fmin(a3[isb3[i]], sbpar2 * a3[i])))) +
fmin(0.0, fmax(sbpar1 * a3[isb3[i]], fmin(a3[i], fmax(a3[isb3[i]], sbpar2 * a3[i]))))) *
(sgn3[i] - dtdx * eiglam3[i]));
}
}

// calculate the final fluxes
if (i > 0) {
d_f1[i] = 0.5 * (fl1[i] + fr1[i] + ac21[i] + ac22[i] + ac23[i]);
d_f2[i] = 0.5 * (fl2[i] + fr2[i] + eiglam1[i] * ac21[i]
+ eiglam2[i] * ac22[i] + eiglam3[i] * ac23[i]);
d_f3[i] = 0.5 * (fl3[i] + fr3[i] + (htilde[i] - utilde[i] * vsc[i]) * ac21[i]
+ absvt[i] * ac22[i] + (htilde[i] + utilde[i] * vsc[i]) * ac23[i]);
}

__syncthreads(); // because of the [i + 1] index below
// update U
if (i > 0 && i < nbrOfGrids - 1) {
d_u1[i] -= *d_tau / *d_h * (d_f1[i + 1] - d_f1[i]);
d_u2[i] -= *d_tau / *d_h * (d_f2[i + 1] - d_f2[i]);
d_u3[i] -= *d_tau / *d_h * (d_f3[i + 1] - d_f3[i]);
}

d_boundaryCondition(nbrOfGrids, d_u1, d_u2, d_u3);
}
}