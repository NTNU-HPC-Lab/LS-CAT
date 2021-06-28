#ifndef MFCC_H 
#define MFCC_H

#include<string>
class feature{
public:
    double x[38];
};


class mfcc{
public:
    int N,tmp;
    feature *features;
     mfcc();
     mfcc(std::string file_name);
};
__host__ __device__ double euclids(feature ,feature );
#endif