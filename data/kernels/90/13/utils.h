#ifndef _UITL_H
#define _UITL_H

#include <cstdlib>
#include <sys/time.h>
#include <iostream>
#include <cmath>

#ifdef TIME_FUNC
#define timeFunc(func, message) \
    gettimeofday(&_start_time, NULL); \
    (func);                          \
    gettimeofday(&_end_time, NULL);   \
    if(_end_time.tv_sec == _start_time.tv_sec){    \
        std::cout << message << " : "   \
                  << (_end_time.tv_usec - _start_time.tv_usec) / 1000.0  \
                  << "ms" << std::endl; \
    }else{  \
        std::cout << message << " : "   \
                  << _end_time.tv_sec - _start_time.tv_sec + \
                     (_end_time.tv_usec - _start_time.tv_usec) / 1000000.0  \
                  << "s" << std::endl; \
    }
#else
#define timeFunc(func, message) (func);                          
#endif

inline float random_float(float low, float upper){
    return (rand() * 1.0 / RAND_MAX) * (upper - low) + low;
}

inline bool float_equal(float a, float b, float e){
    float diff = a - b;
    return (diff < e) && (-diff < e);
}

inline float logisitc(float a){
    return 1.0 / (1 + exp(-a));
}

#endif
