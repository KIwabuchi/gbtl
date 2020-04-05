/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2018 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#ifndef GB_ALGEBRA_HPP
#define GB_ALGEBRA_HPP

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace GraphBLAS
{
    namespace detail
    {
        // Overload Abs for different types (call the right library func).
        template<typename D2>
        inline D2 MyAbs(int8_t input)        { return abs(input); }
        template<typename D2>
        inline D2 MyAbs(int16_t input)       { return abs(input); }
        template<typename D2>
        inline D2 MyAbs(int32_t input)       { return abs(input); } // labs>
        template<typename D2>
        inline D2 MyAbs(int64_t input)       { return labs(input); } // llabs?

        template<typename D2>
        inline D2 MyAbs(float input)         { return fabsf(input); }

        template<typename D2>
        inline D2 MyAbs(double input)        { return fabs(input); }

        // all other types are unsigned; i.e., no op.
        template<typename D2>
        inline D2 MyAbs(bool input)          { return input; }
        template<typename D2>
        inline D2 MyAbs(uint8_t input)       { return input; }
        template<typename D2>
        inline D2 MyAbs(uint16_t input)      { return input; }
        template<typename D2>
        inline D2 MyAbs(uint32_t input)      { return input; }
        template<typename D2>
        inline D2 MyAbs(uint64_t input)      { return input; }

    } // namespace detail (within GraphBLAS namespace

    //************************************************************************
    // The Unary Operators
    //************************************************************************
    // Following the removal of std::unary_function from C++17 and beyond,
    // these functors do not need to subclass from unary_function nor
    // define:
    //   - argument_type
    //   - result_type

    // Also performs casting
    template <typename D1, typename D2 = D1>
    struct Identity
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) const { return input; }
    };

    template <typename D1, typename D2 = D1>
    struct Abs
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) const
        {
            return detail::MyAbs<D2>(input);
        }
    };


    template <typename D1, typename D2 = D1>
    struct AdditiveInverse
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) const { return -input; }
    };

    template <typename D1, typename D2 = D1>
    struct MultiplicativeInverse
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) const
        {
            return static_cast<D2>(1) / input;
        }
    };


    template <typename D1 = bool, typename D2 = D1>
    struct LogicalNot
    {
        typedef D2 result_type;
        inline D2 operator()(D1 input) const { return !input; }
    };

    template <typename I1 = uint64_t, typename I2 = I1>
    struct BitwiseNot
    {
        typedef I2 result_type;
        inline I2 operator()(I1 input) const { return ~input; }
    };

    //************************************************************************
    // User std::bind to turn binary ops into unary ops
    //
    // Turn a binary op into a unary op by binding the 2nd term to a constant
    //
    //                     std::bind(GraphBLAS::Minus<float>(),
    //                               std::placeholders::_1,
    //                               static_cast<float>(nsver)),
    //
    // Turn a binary op into a unary op by binding the 1st term to a constant
    //
    //                     std::bind(GraphBLAS::Minus<float>(),
    //                               static_cast<float>(nsver),
    //                               std::placeholders::_2),
    //
    //************************************************************************
}

namespace GraphBLAS
{
    //************************************************************************
    // The Binary Operators
    //************************************************************************
    // Following the removal of std::binary_function from C++17 and beyond,
    // these functors do not need to subclass from binary_function but should
    // still define:
    //   - first_argument_type
    //   - second_argument_type
    //   - result_type
    //
    // In lambda speak
    // [](auto x, auto y) → D3 { return x * y };
    // [](D1 x, D2 y) -> D3 { return x * y; }

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalOr
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs || rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalAnd
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs && rhs; }
    };

    template <typename D1 = bool, typename D2 = D1, typename D3 = D1>
    struct LogicalXor
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        // ((bool)lhs) != ((bool)rhs)
        // inline D3 operator()(D1 lhs, D2 rhs) const { return lhs ^ rhs; }
        inline D3 operator()(D1 lhs, D2 rhs) const
        {
            return ((lhs && !rhs) || (!lhs && rhs));
        }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct Equal
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs == rhs; }
    };

    template <typename D1, typename D2 = D1, typename D3 = bool>
    struct NotEqual
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs != rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterThan
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs > rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessThan
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct GreaterEqual
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs >= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = bool>
    struct LessEqual
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs <= rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct First
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Second
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Min
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs ? lhs : rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Max
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs < rhs ? rhs : lhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Plus
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs + rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Minus
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs - rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Times
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs * rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Div
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return lhs / rhs; }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Power
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return std::pow(lhs, rhs); }
    };

    template<typename D1, typename D2 = D1, typename D3 = D1>
    struct Xor
    {
        typedef D1 first_argument_type;
        typedef D2 second_argument_type;
        typedef D3 result_type;
        inline D3 operator()(D1 lhs, D2 rhs) const { return (lhs ^ rhs); }
    };

} // namespace GraphBLAS


typedef GraphBLAS::LogicalOr<bool>    GrB_LOR;
typedef GraphBLAS::LogicalAnd<bool>   GrB_LAND;
typedef GraphBLAS::LogicalXor<bool>   GrB_LXOR;

//****************************************************************************
// Monoids
//****************************************************************************

#define GEN_GRAPHBLAS_MONOID(M_NAME, BINARYOP, IDENTITY)        \
    template <typename ScalarT>                                 \
    struct M_NAME                                               \
    {                                                           \
    public:                                                     \
        typedef ScalarT argument_type;                          \
        typedef ScalarT result_type;                            \
                                                                \
        ScalarT identity() const                                \
        {                                                       \
            return static_cast<ScalarT>(IDENTITY);              \
        }                                                       \
                                                                \
        ScalarT operator()(ScalarT lhs, ScalarT rhs) const      \
        {                                                       \
            return BINARYOP<ScalarT>()(lhs, rhs);               \
        }                                                       \
    };

namespace GraphBLAS
{
    GEN_GRAPHBLAS_MONOID(PlusMonoid, Plus, 0)
    GEN_GRAPHBLAS_MONOID(TimesMonoid, Times, 1)
    GEN_GRAPHBLAS_MONOID(MinMonoid, Min, std::numeric_limits<ScalarT>::max())

    /// @todo The following identity only works for unsigned domains
    /// std::numerical_limits<>::min() does not work for floating point types
    GEN_GRAPHBLAS_MONOID(MaxMonoid, Max, 0)

    GEN_GRAPHBLAS_MONOID(LogicalOrMonoid, LogicalOr, false)
} // GraphBLAS

//****************************************************************************
// Semirings
//****************************************************************************

/**
 * The macro for building semi-ring objects
 *
 * @param[in]  SRNAME        The class name
 * @param[in]  ADD_MONOID    The addition monoid
 * @param[in]  MULT_BINARYOP The multiplication binary function
 */
#define GEN_GRAPHBLAS_SEMIRING(SRNAME, ADD_MONOID, MULT_BINARYOP)       \
    template <typename D1, typename D2=D1, typename D3=D1>              \
    class SRNAME                                                        \
    {                                                                   \
    public:                                                             \
        typedef D1 first_argument_type;                                 \
        typedef D2 second_argument_type;                                \
        typedef D3 result_type;                                         \
                                                                        \
        D3 add(D3 a, D3 b) const                                        \
        { return ADD_MONOID<D3>()(a, b); }                              \
                                                                        \
        D3 mult(D1 a, D2 b) const                                       \
        { return MULT_BINARYOP<D1,D2,D3>()(a, b); }                     \
                                                                        \
        D3 zero() const                                                 \
        { return ADD_MONOID<D3>().identity(); }                         \
    };


namespace GraphBLAS
{
    GEN_GRAPHBLAS_SEMIRING(ArithmeticSemiring, PlusMonoid, Times)

    GEN_GRAPHBLAS_SEMIRING(LogicalSemiring, LogicalOrMonoid, LogicalAnd)

    /// @note the Plus operator would need to be "infinity aware" if the caller
    /// were to pass "infinity" sentinel as one of the arguments. But no
    /// GraphBLAS operations 'should' do that.
    GEN_GRAPHBLAS_SEMIRING(MinPlusSemiring, MinMonoid, Plus)

    GEN_GRAPHBLAS_SEMIRING(MaxTimesSemiring, MaxMonoid, Times)

    GEN_GRAPHBLAS_SEMIRING(MinSelect2ndSemiring, MinMonoid, Second)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect2ndSemiring, MaxMonoid, Second)

    GEN_GRAPHBLAS_SEMIRING(MinSelect1stSemiring, MinMonoid, First)
    GEN_GRAPHBLAS_SEMIRING(MaxSelect1stSemiring, MaxMonoid, First)
} // namespace GraphBLAS

//****************************************************************************
// Convert Semirings to BinaryOps
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template <typename SemiringT>
    struct MultiplicativeOpFromSemiring
    {
    public:
        typedef typename SemiringT::first_argument_type  first_argument_type;
        typedef typename SemiringT::second_argument_type second_argument_type;
        typedef typename SemiringT::result_type          result_type;

        MultiplicativeOpFromSemiring() = delete;
        MultiplicativeOpFromSemiring(SemiringT const &sr) : sr(sr) {}

        result_type operator() (first_argument_type lhs,
                                second_argument_type rhs) const
        {
            return sr.mult(lhs, rhs);
        }

    private:
        SemiringT sr;
    };

    //************************************************************************
    template <typename SemiringT>
    struct AdditiveMonoidFromSemiring
    {
    public:
        typedef typename SemiringT::result_type result_type;
        typedef typename SemiringT::result_type argument_type;

        AdditiveMonoidFromSemiring() = delete;
        AdditiveMonoidFromSemiring(SemiringT const &sr) : sr(sr) {}

        argument_type identity() const
        {
            return sr.zero();
        }

        result_type operator() (argument_type lhs, argument_type rhs) const
        {
            return sr.add(lhs, rhs);
        }

    private:
        SemiringT sr;
    };

    //************************************************************************
    template <typename SemiringT>
    MultiplicativeOpFromSemiring<SemiringT>
    multiply_op(SemiringT const &sr)
    {
        return MultiplicativeOpFromSemiring<SemiringT>(sr);
    }

    //************************************************************************
    template <typename SemiringT>
    AdditiveMonoidFromSemiring<SemiringT>
    add_monoid(SemiringT const &sr)
    {
        return AdditiveMonoidFromSemiring<SemiringT>(sr);
    }

} // namespace GraphBLAS

#endif // GB_ALGEBRA_HPP
