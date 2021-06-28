__device__ void SweXMatrix(double *a_x, double *b_x, double *c_x, double *d_x, 
    int N, int j, int t, double *Hs, double *h, double *mann, double *eRF,
    double *IN, double *ET, double *K2w, double *K2e);

__device__ void SweYMatrix(double *a_y, double *b_y, double *c_y, double *d_y,
    int M, int N, int i, int t, double *Hs, double *h, double *mann, 
    double *eRF, double *IN, double *ET, double *K2n, double *K2s);
