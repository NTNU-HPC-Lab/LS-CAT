#include "includes.h"
__device__ void vanGenuchten(double *C, double *K, double *Ksat, double *theta, double *h, double n, double m, int size) {
double Se, h_, theta_;

for (int i = 0; i < size; i++) {
// Convert pressure unit from [m] to [cm]
h_ = h[i] * 100;

// Compute the volumetric moisture content [eqn 21]
if (h_ < 0) {
// for unsaturated soil conditions
theta_ = (theta_S - theta_R)/pow(1.0 + pow((-h_*alpha), n), m) + theta_R;
} else {
// for saturated soil condition h_ >= 0
theta_ = theta_S;
}
theta[i] = theta_;

// Compute the effective saturation [eqn 2]
Se = (theta_ - theta_R)/(theta_S - theta_R);

// Compute the hydraulic conductivity [eqn 8] - [Convert to unit: m/hr
K[i] = Ksat[i] * sqrt(Se) * (1.0 - pow(1.0-pow(Se, 1.0/m), m))
* (1.0 - pow(1.0-pow(Se, 1.0/m), m));

// Compute the specific moisture storage derivative of eqn (21).
// So we have to calculate C = d(theta)/dh. Then the unit is converted
// into [1/m].
if (h_ < 0) {
C[i] = 100 * -alpha * n * -1 * (1.0/n-1.0)*pow(alpha*abs(h_), n-1)
* (theta_R-theta_S) * pow(pow(alpha*abs(h_), n)+1, 1.0/n-2.0);
} else {
C[i] = 0.0;
}
}
}
__global__ void vanGenuchtenIntial(double *theta, double *K, double *Ksat, double *h, int size) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < size) {
double Se, _theta, _h, n, m, lambda;
n = nv;
lambda = n - 1.0;
m = lambda/n;

// Convert unit from [m] to [cm]
_h = h[i] * 100;

// Compute the volumetric moisture content [eqn 21]
if (_h < 0) {
_theta = (theta_S-theta_R) / pow(1.0+pow((alpha*(-_h)), n), m) + theta_R;
} else {
_theta = theta_S;
}
theta[i] = _theta;

// Compute the effective saturation [eqn 2]
Se = (_theta - theta_R)/(theta_S - theta_R);

// Compute the hydraulic conductivity [eqn 8] - Convert to unit: m/hr
K[i] = Ksat[i] * sqrt(Se) * (1.0 - pow(1.0-pow(Se, 1.0/m), m))
*(1.0 - pow(1.0 - pow(Se, 1.0/m), m));

// Update threads if vector is long
i += blockDim.x * gridDim.x;
}
}