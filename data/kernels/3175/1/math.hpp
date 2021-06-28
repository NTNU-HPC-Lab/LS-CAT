/*
** Projeto de Algoritmos Paralelos
** Cálculos Matemáticos
*/

#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

namespace Math {
    inline double* product(double x, double* v, long n) {
        double* result = new double[n];

        for (long i = 0; i < n; i++)
            result[i] = x * v[i];

        return result;
    }

    inline void sub(double* v1, double* v2, long n) {
        for (long i = 0; i < n; i++)
            v1[i] = v1[i] - v2[i];
        delete[] v2;
    }

    inline double norm(double* v, long n) {
        double sum = 0;

        for (long i = 0; i < n; i++)
            sum += (v[i] * v[i]);
            
        return sqrt(sum);
    }

    inline void normalize(double* v, long n) {
        double nm = norm(v, n);

        for (long i = 0; i < n; i++)
            v[i] /= nm;
    }

    inline static double dotProduct(double* v1, double* v2, long n) {
        double sum = 0;

        for (long i = 0; i < n; i++)
            sum += (v1[i] * v2[i]);

        return sum;
    }

    inline double consineSimilarity(double* v1, double* v2, int n) {
        return dotProduct(v1, v2, n) / (norm(v1, n) * norm(v2, n));
    }
}

#endif //MATH_HPP