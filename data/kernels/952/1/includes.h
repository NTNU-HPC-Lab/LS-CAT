#define RAD 1
const int nThreads = 61; // can divide NX
const unsigned int NX = 122; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int NY = 101; // number of grid points in y-direction, meaning NY-1 cells
 __constant__ double dx = 1.0 / 100.0; //need to change according to NX and LX
 __constant__ double voltage = 1.0e4;
 __constant__ double eps = 1.0e-4;
//new series 
