const unsigned int NX = 200; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int NY = 8; // number of grid points in y-direction, meaning NY-1 cells
const unsigned int NZ = 101;
 __constant__ double CFL = 0.01; // CFL = dt/dx
 __constant__ double cs_square = 1.0 / 3.0 / (0.01*0.01); // 1/3/(CFL^2)
 __constant__ double K = 2.5e-5;
 __constant__ double w0  = 8.0 / 27.0;  // zero weight for i=0
 __constant__ double ws  = 2.0 / 27.0;  // adjacent weight for i=1-6
 __constant__ double wa  = 1.0 / 54.0;  // adjacent weight for i=7-18
 __constant__ double wd  = 1.0 / 216.0; // diagonal weight for i=19-26
//new series 
