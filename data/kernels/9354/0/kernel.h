__global__ void addKernel(double *c, const double *a, const double *b, int Width);
__global__ void MatrixMulKernel(double *OutMat, double *Mat1, double *Mat2,  int Aheight, int Awidth, int Bwidth);
__global__ void MatrixSquareKernel(double *OutMat, double *Mat1, int width);
