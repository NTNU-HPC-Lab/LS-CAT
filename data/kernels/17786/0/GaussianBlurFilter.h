#ifndef GBF_H
#define GBF_H
class GBFilter
{
public:
    double **getGaussian(int, int, double);
    double **applyFilter(double **, double **, int, int);
};
#endif