#pragma once

/**

Arbitrary sized vector. Vector is column vector (N x 1 matrix)
which means that it can only be multipled with matrices from right.

N should be 2, 3 or 4 at most.

*/

#include <algorithm>
#include <type_traits>
#include <tuple>
#include <cmath>
#include "CudaCheck.h"

template<class T>
using ArithmeticEnable = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template<class T>
using ArithmeticEnable = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template<class... Args>
using AllArithmeticEnable = typename std::enable_if<std::conjunction<std::is_arithmetic<Args>...>::value>::type;

template<class T, class RType = void>
using FloatEnable = typename std::enable_if<std::is_floating_point<T>::value, RType>::type;

template<class T, class RType = void>
using IntegralEnable = typename std::enable_if<std::is_integral<T>::value, RType>::type;

template<class T, class RType = void>
using SignedEnable = typename std::enable_if<std::is_signed<T>::value, RType>::type;

template<int N, class T, typename = ArithmeticEnable<T>>
class Vector;

static constexpr size_t ChooseVectorAlignment(size_t totalSize)
{
    if(totalSize <= 4)
        return 4;           // 1byte Vector Types
    else if(totalSize <= 8)
        return 8;           // 4byte Vector2 Types
    else if(totalSize < 16)
        return 4;           // 4byte Vector3 Types
    else
        return 16;          // 4byte Vector4 Types
}

template<int N, class T>
class alignas(ChooseVectorAlignment(N * sizeof(T))) Vector<N, T>
{
    static_assert(N == 2 || N == 3 || N == 4, "Vector size should be 2, 3 or 4");

    private:
        T                                   vector[N];

    protected:
    public:
        // Constructors & Destructor
        constexpr                           Vector() = default;
        template<class C, typename = ArithmeticEnable<C>>
        __device__ __host__                 Vector(C);
        template<class C, typename = ArithmeticEnable<C>>
        __device__ __host__                 Vector(const C* data);
        template <class... Args, typename = AllArithmeticEnable<Args...>>
        constexpr __device__ __host__       Vector(const Args... dataList);
        template <class... Args, typename = std::enable_if_t<((N - sizeof...(Args)) > 1)>>
        __device__ __host__                 Vector(const Vector<N - sizeof...(Args), T>&,
                                                   const Args... dataList);
        template <int M, typename = std::enable_if_t<(M > N)>>
        __device__ __host__                 Vector(const Vector<M, T>&);
                                            ~Vector() = default;

        // MVC bug? these trigger std::trivially_copyable static assert
        // __device__ __host__              Vector(const Vector&) = default;
        // __device__ __host__ Vector&      operator=(const Vector&) = default;

        // Accessors
        __device__ __host__ explicit            operator T*();
        __device__ __host__ explicit            operator const T*() const;
        __device__ __host__ T&                  operator[](int);
        __device__ __host__ constexpr const T&  operator[](int) const;

        // Type cast
        template<int M, class C, typename = std::enable_if_t<(M <= N)>>
        __device__ __host__ explicit                    operator Vector<M, C>() const;

        // Modify
        __device__ __host__ void                        operator+=(const Vector&);
        __device__ __host__ void                        operator-=(const Vector&);
        __device__ __host__ void                        operator*=(const Vector&);
        __device__ __host__ void                        operator*=(T);
        __device__ __host__ void                        operator/=(const Vector&);
        __device__ __host__ void                        operator/=(T);

        __device__ __host__ Vector                      operator+(const Vector&) const;
        __device__ __host__ Vector                      operator-(const Vector&) const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector>     operator-() const;
        __device__ __host__ Vector                      operator*(const Vector&) const;
        __device__ __host__ Vector                      operator*(T) const;
        __device__ __host__ Vector                      operator/(const Vector&) const;
        __device__ __host__ Vector                      operator/(T) const;

        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      operator%(const Vector&) const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      operator%(T) const;
        template<class Q = T>
        __device__ __host__ IntegralEnable<Q, Vector>   operator%(const Vector&) const;
        template<class Q = T>
        __device__ __host__ IntegralEnable<Q, Vector>   operator%(T) const;

        // Logic
        __device__ __host__ bool                        operator==(const Vector&) const;
        __device__ __host__ bool                        operator!=(const Vector&) const;
        __device__ __host__ bool                        operator<(const Vector&) const;
        __device__ __host__ bool                        operator<=(const Vector&) const;
        __device__ __host__ bool                        operator>(const Vector&) const;
        __device__ __host__ bool                        operator>=(const Vector&) const;

        // Utilty
        __device__ __host__ T                           Dot(const Vector&) const;

        template<class Q = T>
        __device__ __host__ FloatEnable<Q, T>           Length() const;
        __device__ __host__ T                           LengthSqr() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Normalize() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     NormalizeSelf();
        __device__ __host__ Vector                      Clamp(const Vector&, const Vector&) const;
        __device__ __host__ Vector                      Clamp(T min, T max) const;
        __device__ __host__ Vector&                     ClampSelf(const Vector&, const Vector&);
        __device__ __host__ Vector&                     ClampSelf(T min, T max);

        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector>     Abs() const;
        template<class Q = T>
        __device__ __host__ SignedEnable<Q, Vector&>    AbsSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Round() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     RoundSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Floor() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     FloorSelf();
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector>      Ceil() const;
        template<class Q = T>
        __device__ __host__ FloatEnable<Q, Vector&>     CeilSelf();

        static __device__ __host__ Vector               Min(const Vector&, const Vector&);
        static __device__ __host__ Vector               Min(const Vector&, T);
        static __device__ __host__ Vector               Max(const Vector&, const Vector&);
        static __device__ __host__ Vector               Max(const Vector&, T);

        template<class Q = T>
        static __device__ __host__ FloatEnable<Q, Vector>   Lerp(const Vector&,
                                                                 const Vector&,
                                                                 T);
};

// Left scalars
template<int N, class T>
static __device__ __host__ Vector<N,T> operator*(T, const Vector<N, T>&);

// Typeless vectors are defaulted to float
using Vector2 = Vector<2, float>;
using Vector3 = Vector<3, float>;
using Vector4 = Vector<4, float>;
// Float Type
using Vector2f = Vector<2, float>;
using Vector3f = Vector<3, float>;
using Vector4f = Vector<4, float>;
// Double Type
using Vector2d = Vector<2, double>;
using Vector3d = Vector<3, double>;
using Vector4d = Vector<4, double>;
// Integer Type
using Vector2i = Vector<2, int>;
using Vector3i = Vector<3, int>;
using Vector4i = Vector<4, int>;
// Unsigned Integer Type
using Vector2ui = Vector<2, unsigned int>;
using Vector3ui = Vector<3, unsigned int>;
using Vector4ui = Vector<4, unsigned int>;
// Long Types
using Vector2l = Vector<2, int64_t>;
using Vector2ul = Vector<2, uint64_t>;

// Requirements of Vectors
//static_assert(std::is_literal_type<Vector3>::value == true, "Vectors has to be literal types");
static_assert(std::is_trivially_copyable<Vector3>::value == true, "Vectors has to be trivially copyable");
static_assert(std::is_polymorphic<Vector3>::value == false, "Vectors should not be polymorphic");

// Alignment Checks
static_assert(sizeof(Vector2) == 8, "Vector2 should be tightly packed");
static_assert(sizeof(Vector3) == 12, "Vector3 should be tightly packed");
static_assert(sizeof(Vector4) == 16, "Vector4 should be tightly packed");

// Cross product (only for 3d vectors)
template <class T>
static __device__ __host__ Vector<3, T> Cross(const Vector<3, T>&, const Vector<3, T>&);

// Implementation
#include "Vector.hpp"   // CPU & GPU

// Basic Constants
static constexpr Vector3 XAxis = Vector3(1.0f, 0.0f, 0.0f);
static constexpr Vector3 YAxis = Vector3(0.0f, 1.0f, 0.0f);
static constexpr Vector3 ZAxis = Vector3(0.0f, 0.0f, 1.0f);

// Zero Constants
static constexpr Vector2 Zero2f = Vector2(0.0f, 0.0f);
static constexpr Vector3 Zero3f = Vector3(0.0f, 0.0f, 0.0f);
static constexpr Vector4 Zero4f = Vector4(0.0f, 0.0f, 0.0f, 0.0f);

static constexpr Vector2 Zero2d = Vector2(0.0, 0.0);
static constexpr Vector3 Zero3d = Vector3(0.0, 0.0, 0.0);
static constexpr Vector4 Zero4d = Vector4(0.0, 0.0, 0.0, 0.0);

static constexpr Vector2i Zero2i = Vector2i(0, 0);
static constexpr Vector3i Zero3i = Vector3i(0, 0, 0);
static constexpr Vector4i Zero4i = Vector4i(0, 0, 0, 0);

static constexpr Vector2ui Zero2ui = Vector2ui(0u, 0u);
static constexpr Vector3ui Zero3ui = Vector3ui(0u, 0u, 0u);
static constexpr Vector4ui Zero4ui = Vector4ui(0u, 0u, 0u, 0u);

static constexpr Vector2ul Zero2ul = Vector2ul(0ul, 0ul);

static constexpr Vector2 Zero2 = Zero2f;
static constexpr Vector3 Zero3 = Zero3f;
static constexpr Vector4 Zero4 = Zero4f;

// Vector Traits
template<class T>
struct IsVectorType
{
    static constexpr bool value =
        std::is_same<T, Vector2f>::value ||
        std::is_same<T, Vector2d>::value ||
        std::is_same<T, Vector2i>::value ||
        std::is_same<T, Vector2ui>::value ||
        std::is_same<T, Vector3f>::value ||
        std::is_same<T, Vector3d>::value ||
        std::is_same<T, Vector3i>::value ||
        std::is_same<T, Vector3ui>::value ||
        std::is_same<T, Vector4f>::value ||
        std::is_same<T, Vector4d>::value ||
        std::is_same<T, Vector4i>::value ||
        std::is_same<T, Vector4ui>::value;
};

// Vector Etern
extern template class Vector<2, float>;
extern template class Vector<2, double>;
extern template class Vector<2, int>;
extern template class Vector<2, unsigned int>;

extern template class Vector<3, float>;
extern template class Vector<3, double>;
extern template class Vector<3, int>;
extern template class Vector<3, unsigned int>;

extern template class Vector<4, float>;
extern template class Vector<4, double>;
extern template class Vector<4, int>;
extern template class Vector<4, unsigned int>;

extern template class Vector<2, int64_t>;
extern template class Vector<2, uint64_t>;