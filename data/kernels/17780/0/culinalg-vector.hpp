/**
 * Class and methods for representing a large dimensional vector.
 */

#ifndef CULINALG_HEADER_VECTOR
#define CULINALG_HEADER_VECTOR

#include <cstddef>

namespace clg
{
    /*
     * Forward declaration of CuData
     */
    class CuData;

    // TODO add double precision support
    /**
     * A very large dimensional vector with operators overloaded for vector addition.
     */
    class Vector                                    
    {
        public:
            /**
             * Construct a vector of dimensionality n. If init is true, initialize vector to 0.
             * Strong exception guarantee.
             */
            Vector(size_t n, bool init = true);
            /**
             * Destruct a vector. Strong exception guarantee.
             */
            ~Vector();
            /**
             * Copy constructor. Strong exception guarantee.
             */
            Vector(const Vector& other);
            /**
             * Move constructor. Strong exception guarantee.
             */
            Vector(Vector&& other);
            /**
             * Copy assignment operator. Strong exception guarantee.
             */
            Vector& operator=(const Vector& other);
            /**
             * Move assignment operator. Strong exception guarantee.
             */
            Vector& operator=(Vector&& other);
            /**
             * Friend function for vector addition.
             */
            friend Vector operator+(const Vector& x, const Vector& y);
            /**
             * Compound operator for vector addition. Strong exception guarantee.
             */
            Vector& operator+=(const Vector& other);
            /**
             * Access an element of a vector as a float. Strong exception guarantee.
             */
            float& operator[](size_t index);
        
        private:
            // Must point to valid CuData at all times. The CuData should point to some data, unless
            // recovering from a recent exception.
            CuData* irepr_;
            const size_t dim_;

            // Attempt to allocate data in irepr_ . Throws exceptio on allocation failure. Strong
            // exception guarantee. Assumes inviariants of CuData hold.
            void alloc_irepr_throw_();
            // Attempt to delete all data in irepr_, throw exceptions on failure. Should not be used
            // in dtor. Strong exception guarantee.
            void delloc_irepr_throw_();

    };

    /**
     * Adds two vectors together. Friend to Vector. Strong exception guarantee. 
     * @param x The left argument
     * @param y The right argument
     */
    Vector operator+(const Vector& x, const Vector& y);
}

#endif
