__device__ int perturb = 0;// Change to 1 to apply finite amplitude perturbation
const unsigned int NX = 122; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int NY = 101; // number of grid points in y-direction, meaning NY-1 cells
 __constant__ double charge0 = 10.0;
 __constant__ double w0 = 4.0 / 9.0;  // zero weight
 __constant__ double ws = 1.0 / 9.0;  // adjacent weight
 __constant__ double wd = 1.0 / 36.0; // diagonal weight
//new series 
