__device__ void RichZMatrix(double *a, double *b, double *c, double *d,
    double *Cnp1m, double *Knp1m, double *K_prev, double *hnp1m, double *h_prev,
    double *thetanp1m, double *thetan, int i, int j, int M, int N, int P);


__device__ void RichXMatrix(double *a, double *b, double *c, double *d,
    double *Cnp1m, double *Knp1m, double *K_prev, double *hnp1m, double *h_prev,
    double *thetanp1m, double *thetan, int j, int k, int M, int N, int P);

__device__ void RichYMatrix(double *a, double *b, double *c, double *d, 
    double *Cnp1m, double *Knp1m, double *K_prev, double *hnp1m, double *h_prev,
    double *thetanp1m, double *thetan, int i, int k, int M, int N, int P);