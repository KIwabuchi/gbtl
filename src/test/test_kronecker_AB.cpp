/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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

//#define GRAPHBLAS_LOGGING_LEVEL 2

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE kronecker_test_suite

#include <boost/test/included/unit_test.hpp>

using namespace GraphBLAS;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

namespace
{
    static std::vector<std::vector<double> > A_sparse_3x3 =
    {{1,  0,  0},
     {0,  2,  0},
     {3,  0,  4}};

    static std::vector<std::vector<double> > Ar_sparse_3x3 =
    {{1,  0,  0},
     {0,  2,  0},
     {0,  0,  0}};

    static std::vector<std::vector<double> > Ac_sparse_3x3 =
    {{0,  0,  0},
     {0,  2,  0},
     {0,  0,  4}};

    static std::vector<std::vector<double> > AT_sparse_3x3 =
    {{1,  0,  3},
     {0,  2,  0},
     {0,  0,  4}};

    static std::vector<std::vector<double> > B_sparse_3x4 =
    {{1,  1,  0,  0},
     {0,  2,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > Br_sparse_3x4 =
    {{0,  0,  0,  0},
     {0,  2,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > Bc_sparse_3x4 =
    {{1,  0,  0,  0},
     {0,  0,  2,  0},
     {3,  0,  0,  3}};

    static std::vector<std::vector<double> > BT_sparse_3x4 =
    {{1,  0,  3},
     {1,  2,  0},
     {0,  2,  0},
     {0,  0,  3}};

    static std::vector<std::vector<double> > Answer_rr_sparse_9x12 =
    {{0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}
    };

    static std::vector<std::vector<double> > Answer_sparse_9x12 =
    {{1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  2, 2, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {3, 3, 0, 0,  0, 0, 0, 0,  4, 4, 0, 0},
     {0, 6, 6, 0,  0, 0, 0, 0,  0, 8, 8, 0},
     {9, 0, 0, 9,  0, 0, 0, 0, 12, 0, 0,12}
    };

    static std::vector<std::vector<double> > Answer_rc_sparse_9x12 =
    {{1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {3, 0, 0, 3,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}
    };

    static std::vector<std::vector<double> > Answer_cr_sparse_9x12 =
    {{0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  6, 0, 0, 6,  0, 0, 0, 0},

     {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
     {0, 0, 0, 0,  0, 0, 0, 0,  0, 8, 8, 0},
     {0, 0, 0, 0,  0, 0, 0, 0, 12, 0, 0,12}
    };

    static std::vector<std::vector<double> > Ones_4x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_3x4 =
    {{1, 1, 1, 1},
     {1, 1, 1, 1},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_3x3 =
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}};

    static std::vector<std::vector<double> > Ones_9x9 =
    {{1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > Ones_9x12 =
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > Identity_3x3 =
    {{1, 0, 0},
     {0, 1, 0},
     {0, 0, 1}};

    static std::vector<std::vector<double> > Lower_3x3 =
    {{1, 0, 0},
     {1, 1, 0},
     {1, 1, 1}};

    static std::vector<std::vector<double> > Lower_3x4 =
    {{1, 0, 0, 0},
     {1, 1, 0, 0},
     {1, 1, 1, 0}};

    static std::vector<std::vector<double> > Lower_4x4 =
    {{1, 0, 0, 0},
     {1, 1, 0, 0},
     {1, 1, 1, 0},
     {1, 1, 1, 1}};

    static std::vector<std::vector<double> > NotLower_3x3 =
    {{0, 1, 1},
     {0, 0, 1},
     {0, 0, 0}};

    static std::vector<std::vector<double> > NotLower_3x4 =
    {{0, 1, 1, 1},
     {0, 0, 1, 1},
     {0, 0, 0, 1}};

    static std::vector<std::vector<double> > NotLower_4x4 =
    {{0, 1, 1, 1},
     {0, 0, 1, 1},
     {0, 0, 0, 1},
     {0, 0, 0, 0}};

    static std::vector<std::vector<double> > LowerMask_3x4 =
    {{1, 0,    0,   0},
     {1, 0.5,  0,   0},
     {1, -1.0, 1.5, 0}};

    static std::vector<std::vector<bool> > LowerBool_3x4 =
    {{true, false, false, false},
     {true, true,  false, false},
     {true, true,  true,  false}};

    static std::vector<std::vector<bool> > LowerBool_3x3 =
    {{true, false, false},
     {true, true,  false},
     {true, true,  true}};

    static std::vector<std::vector<bool> > NotLowerBool_3x3 =
    {{false,  true, true},
     {false, false, true},
     {false, false, false}};

    static std::vector<std::vector<double> > M_9x9 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > M_9x12 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

    static std::vector<std::vector<double> > M_10x12 =
        {{1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0},
         {1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
}

//****************************************************************************
// API error tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_bad_dimensions)
{
    Matrix<double> A(A_sparse_3x3, 0.); // 3x3
    Matrix<double> B(B_sparse_3x4, 0.); // 3x4
    Matrix<double> ones9x12(Ones_9x12, 0.);

    Matrix<double> M9x9(M_9x9, 0.);
    Matrix<double> M9x12(M_9x12, 0.);
    Matrix<double> M10x12(M_10x12, 0.);

    Matrix<double> result9x12(9, 12);
    Matrix<double> result9x9(9, 9);
    Matrix<double> result12x12(12, 12);

    // NoMask_NoAccum_AB

    // dim(C) != dim(A)*dim(B)
    BOOST_CHECK_THROW(
        (kronecker(result9x9, NoMask(), NoAccumulate(),
                   Times<double>(), A, B)),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(result12x12, NoMask(), NoAccumulate(),
                   Times<double>(), A, B)),
        DimensionException);

    // NoMask_Accum_AB

    // dim(C) != dim(A)*dim(B)
    BOOST_CHECK_THROW(
        (kronecker(result9x9, NoMask(), Plus<double>(),
                   Times<double>(), A, B)),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(result12x12, NoMask(), Plus<double>(),
                   Times<double>(), A, B)),
        DimensionException);

    // Mask_NoAccum

    // incompatible mask matrix dimensions
    // nrows(C) != nrows(M)
    BOOST_CHECK_THROW(
        (kronecker(result9x12, M10x12, NoAccumulate(),
                   Times<double>(), A, B)),
        DimensionException);

    // ncols(C) != ncols(M)
    BOOST_CHECK_THROW(
        (kronecker(result9x12, M9x9, NoAccumulate(),
                   Times<double>(), A, B)),
        DimensionException);

    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M9x12, NoAccumulate(),
                   Times<double>(), A, A,
                   REPLACE)),
        DimensionException);

    // Mask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M10x12, Second<double>(),
                   Times<double>(), A, B, REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, M9x9, Second<double>(),
                   Times<double>(), A, B, MERGE)),
        DimensionException);

    // CompMask_NoAccum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M10x12), NoAccumulate(),
                   Times<double>(), A, B, REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M9x9), NoAccumulate(),
                   Times<double>(), A, B, MERGE)),
        DimensionException);

    // CompMask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M10x12), Plus<double>(),
                   Times<double>(), A, B, REPLACE)),
        DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (kronecker(ones9x12, complement(M9x9), Plus<double>(),
                   Times<double>(), A, B, MERGE)),
        DimensionException);
}

//****************************************************************************
// NoMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> B(B_sparse_3x4, 0.);

    Matrix<double> answer(Answer_sparse_9x12, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), A, B);

    for (IndexType ix = 0; ix < answer.nrows(); ++ix)
    {
        for (IndexType iy = 0; iy < answer.ncols(); ++iy)
        {
            BOOST_CHECK_EQUAL(C.hasElement(ix, iy), answer.hasElement(ix, iy));
            if (C.hasElement(ix, iy))
            {
                BOOST_CHECK_CLOSE(C.extractElement(ix,iy),
                                  answer.extractElement(ix,iy), 0.0001);
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_empty)
{
    Matrix<double> Zero(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(Ones_9x9, 0.);
    Matrix<double> mD(Ones_9x9, 0.);
    Matrix<double> Zero9x9(9, 9);

    kronecker(C, NoMask(), NoAccumulate(), Times<double>(), Zero, Ones);
    BOOST_CHECK_EQUAL(C, Zero9x9);

    kronecker(mD, NoMask(), NoAccumulate(), Times<double>(), Ones, Zero);
    BOOST_CHECK_EQUAL(mD, Zero9x9);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_dense)
{
    Matrix<double> Ones3x3(Ones_3x3, 0.);
    Matrix<double> Ones3x4(Ones_3x4, 0.);
    Matrix<double> Ones9x12(Ones_9x12, 0.);
    Matrix<double> result(9, 12);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), Ones3x3, Ones3x4);

    BOOST_CHECK_EQUAL(result, Ones9x12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_empty_rows)
{
    Matrix<double> C(9, 12);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_3x4, 0.);

    Matrix<double> answer(Answer_rr_sparse_9x12, 0.);

    kronecker(C, NoMask(), NoAccumulate(),
              Times<double>(), A, B);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_emptyRowA_emptyColB)
{
    Matrix<double> result(9, 12);
    Matrix<double> A(Ar_sparse_3x3, 0.);
    Matrix<double> B(Bc_sparse_3x4, 0.);
    Matrix<double> answer(Answer_rc_sparse_9x12, 0.);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_emptyColA_emptyRowB)
{
    Matrix<double> result(9, 12);
    Matrix<double> A(Ac_sparse_3x3, 0.);
    Matrix<double> B(Br_sparse_3x4, 0.);
    Matrix<double> answer(Answer_cr_sparse_9x12, 0.);

    kronecker(result, NoMask(), NoAccumulate(),
              Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}
#if 0

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    Matrix<double, DirectedMatrixTag> mat(Symmetric_4x4, 0.);
    Matrix<double, DirectedMatrixTag> m3(4, 4);
    Matrix<double, DirectedMatrixTag> answer(Symmetric2_4x4, 0.);

    kronecker(m3,
        NoMask(), NoAccumulate(),
        Times<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_ACdup)
{
    Matrix<double> C(A_sparse_3x3, 0.);
    Matrix<double> B(A_sparse_3x3, 0.);

    Matrix<double> answer(AA_answer_sparse, 0.);

    kronecker(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(), C, B);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_NoAccum_AB_BCdup)
{
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> C(B_sparse_3x4, 0.);

    Matrix<double> answer(Answer_sparse, 0.);

    kronecker(C,
                   NoMask(),
                   NoAccumulate(),
                   Times<double>(), A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// NoMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB)
{
    Matrix<double> A(A_dense_3x3, 0.); // 3x3
    Matrix<double> B(B_dense_3x4, 0.); // 3x4
    Matrix<double> result(3, 4);
    Matrix<double> answer(Answer_dense, 0.);

    kronecker(result,
                   NoMask(),
                   Second<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_empty)
{
    Matrix<double> Zero(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(Ones_3x3, 0.);
    Matrix<double> mD(Ones_3x3, 0.);

    kronecker(C,
                   NoMask(), Plus<double>(),
                   Times<double>(), Zero, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    kronecker(mD,
                   NoMask(), Plus<double>(),
                   Times<double>(), Ones, Zero);
    BOOST_CHECK_EQUAL(mD, Ones);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > B_mat = {{ 1,-2, 0,  0},
                                            {-1, 1, 0,  0},
                                            { 0, 0, 3, -4},
                                            { 0, 0,-3,  3}};
    Matrix<double> A(Symmetric_4x4, 0);
    Matrix<int> B(B_mat, 0);
    Matrix<int> result(4, 4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<std::vector<int> > ans = {{  0,  -1, NIL, NIL},
                                          { -1,   0,   6,  -8},
                                          { -2,   2,   0,  -3},
                                          {NIL, NIL,  -3,   0}};
    Matrix<int> answer(ans, NIL);

    kronecker(result,
                   NoMask(),
                   Second<int>(),
                   Times<int>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_ABdup_Cempty)
{
    // Build some matrices.
    Matrix<double> mat(Symmetric_4x4, 0.);
    Matrix<double> result(4, 4);
    Matrix<double> answer(Symmetric2_4x4, 0.);

    kronecker(result,
                   NoMask(),
                   Second<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    std::vector<std::vector<double>> answer_vals = {{2, 1, 8, 16},
                                                    {1, 1, 1, 1},
                                                    {10,1, 12, 16}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> result(Ones_3x4, 0.);
    Matrix<double> answer(answer_vals, 0.);

    kronecker(result,
                   NoMask(),
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{1, 9, 1, 9},
                                                    {1, 2, 1, 2},
                                                    {1, 5, 1, 5}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> result(Ones_3x4, 0.);
    Matrix<double> answer(answer_vals, 0.);

    kronecker(result,
                   NoMask(),
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_ABdup)
{
    // Build some matrices.
    Matrix<double> mat(A_sparse_3x3,0.);
    Matrix<double> m3(Ones_3x3, 0.);

    // A_sparse_3x3 * A_sparse_3x3 + Ones
    static std::vector<std::vector<double> > ans =
        {{145.,  50., 1},
         {1.0,   26., 1},
         {148.,  50., 82.}};

    Matrix<double> answer(ans, 0.);

    kronecker(m3,
        NoMask(), Plus<double>(),
        Times<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_ACdup)
{
    Matrix<double> C(A_sparse_3x3, 0.);
    Matrix<double> B(A_sparse_3x3, 0.);

    // A_sparse_3x3 * A_sparse_3x3 + A_sparse_3x3
    static std::vector<std::vector<double> > ans =
        {{156.,  56., 0},
         {0.0,   20., 0},
         {154.,  49., 90.}};
    Matrix<double> answer(ans, 0.);

    kronecker(C,
                   NoMask(),
                   Plus<double>(),
                   Times<double>(), C, B);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_NoMask_Accum_AB_BCdup)
{
    Matrix<double> A(A_sparse_3x3, 0.);
    Matrix<double> C(A_sparse_3x3, 0.);

    // A_sparse_3x3 * A_sparse_3x3 + A_sparse_3x3
    static std::vector<std::vector<double> > ans =
        {{156.,  56., 0},
         {0.0,   20., 0},
         {154.,  49., 90.}};
    Matrix<double> answer(ans, 0.);

    kronecker(C,
                   NoMask(),
                   Plus<double>(),
                   Times<double>(), A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

// ****************************************************************************
// Mask_NoAccum
// ****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB)
{
    Matrix<double> A(A_sparse_3x3, 0.0);
    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   Empty, NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   Ones, NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    kronecker(C,
                   A, NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, AFilled);

    C = Ones;
    kronecker(C,
                   MLower, NoAccumulate(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   MNotLower, NoAccumulate(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   Empty, NoAccumulate(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   Ones, NoAccumulate(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    kronecker(C,
                   MLower, NoAccumulate(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    kronecker(C,
                   MNotLower, NoAccumulate(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, MNotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_ABM_empty)
{
    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    Matrix<bool> M(LowerBool_3x3, false);

    Matrix<double> mUpper(NotLower_3x3, 0.);

    // Merge
    C = Ones;
    kronecker(C,
                   M, NoAccumulate(),
                   Times<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    kronecker(C,
                   M, NoAccumulate(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    kronecker(C,
                   Empty, NoAccumulate(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C,
                   M, NoAccumulate(),
                   Times<double>(), Empty, Ones, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   M, NoAccumulate(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   Empty, NoAccumulate(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_Merge_full_mask)
{
    IndexArrayType i_A      =  {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_A      =  {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_A = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 3);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_B      = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_B      = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_B = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> B(3, 4);
    B.build(i_B, j_B, v_B);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    Matrix<unsigned int, DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    mask.build(i_answer, j_answer, v_mask);

    kronecker(result,
        mask, NoAccumulate(),
        Times<double>(),
        A, B);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_mask_not_full)
{
    IndexArrayType i_A    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_A    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_A = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> A(3, 3);
    A.build(i_A, j_A, v_A);

    IndexArrayType i_B    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_B    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_B = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> B(3, 4);
    B.build(i_B, j_B, v_B);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    Matrix<unsigned int, DirectedMatrixTag> mask(3,4);
    std::vector<unsigned int> v_mask(i_answer.size(), 1);
    mask.build(i_answer, j_answer, v_mask);

    kronecker(result,
        mask, NoAccumulate(),
        Times<double>(),
        A, B);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_Merge_Cones_Mlower_stored_zero)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(Lower_3x4, 0.);
    M.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   M,
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> Lower(Lower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11,0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {0, 0, 1, 1},
                                                     {9, 0, 11,1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> Lower(Lower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{0, 0, 0, 0},
                                                    {0, 1, 0, 0},
                                                    {0, 4, 0, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{0, 1, 1, 1},
                                                     {0, 1, 1, 1},
                                                     {0, 4, 0, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};


    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> NotLower(NotLower_3x3, 0.);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 0, 7},
                                                    {0, 0, 0},
                                                    {0, 0, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   NotLower,
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 0, 7},
                                                     {1, 1, 0},
                                                     {1, 1, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   NotLower,
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_ACdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), C, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   NoAccumulate(),
                   Times<double>(), mat, C,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_NoAccum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = Lower;
    kronecker(C,
                   C,
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = Lower;
    kronecker(C,
                   C,
                   NoAccumulate(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
// Mask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB)
{
    Matrix<double> A(A_sparse_3x3, 0.0);
    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   Empty, Plus<double>(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    //---
    static std::vector<std::vector<double> > ans =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    kronecker(C,
                   A, Plus<double>(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans, 0.));

    //---
    static std::vector<std::vector<double> > ans2 =
        {{2,  1,  1},
         {2,  2,  1},
         {2,  2,  2}};

    C = Ones;
    kronecker(C,
                   MLower, Plus<double>(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans2, 0.));

    //---
    static std::vector<std::vector<double> > ans3 =
        {{1,  2,  2},
         {1,  1,  2},
         {1,  1,  1}};

    C = Ones;
    kronecker(C,
                   MNotLower, Plus<double>(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans3, 0.));

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   Empty, Plus<double>(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    //---
    static std::vector<std::vector<double> > ans4 =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    kronecker(C,
                   Ones, Plus<double>(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans4, 0.));

    //---
    static std::vector<std::vector<double> > ans5 =
        {{2,  0,  0},
         {2,  2,  0},
         {2,  2,  2}};

    C = Ones;
    kronecker(C,
                   MLower, Plus<double>(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans5, 0.));

    //---
    static std::vector<std::vector<double> > ans6 =
        {{0,  2,  2},
         {0,  0,  2},
         {0,  0,  0}};

    C = Ones;
    kronecker(C,
                   MNotLower, Plus<double>(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans6, 0.));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_ABMempty)
{
    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    Matrix<bool> M(LowerBool_3x3, false);

    // Merge

    C = Ones;
    kronecker(C,
                   M, Plus<double>(),
                   Times<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   M, Plus<double>(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   Empty, Plus<double>(),
                   Times<double>(), Ones, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C,
                   M, Plus<double>(),
                   Times<double>(), Empty, Ones, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    kronecker(C,
                   M, Plus<double>(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    kronecker(C,
                   Empty, Plus<double>(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> Lower(Lower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{2, 0, 0, 0},
                                                    {1, 1, 0, 0},
                                                    {10,1, 12,0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{2, 1, 1, 1},
                                                     {1, 1, 1, 1},
                                                     {10,1,12,1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> Lower(Lower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {1, 2, 0, 0},
                                                    {1, 5, 1, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {1, 2, 1, 1},
                                                     {1, 5, 1, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};


    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> NotLower(NotLower_3x3, 0.);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 1, 8},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   NotLower,
                   Plus<double>(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 8},
                                                     {1, 1, 1},
                                                     {1, 1, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   NotLower,
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  1,  1},
                                             {4, 10,  1,  1},
                                             {3, 11, 23,  1},
                                             {1,  7, 22, 26}};
    Matrix<double> answer(ans, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    Matrix<double> answer2(ans2, 0.);

    C = Ones;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_ACdup)
{

    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), C, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   Lower,
                   Plus<double>(),
                   Times<double>(), mat, C,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> Lower(Lower_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  0,  0,  0},
                                             {4, 10,  0,  0},
                                             {3, 11, 23,  0},
                                             {1,  7, 22, 26}};
    Matrix<double> answer(ans, 0.);

    C = Lower;
    kronecker(C,
                   C,
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    Matrix<double> answer2(ans2, 0.);

    C = Lower;
    kronecker(C,
                   C,
                   Plus<double>(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_Replace_lower_mask_result_ones)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(LowerMask_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   M,
                   Second<double>(),
                   Times<double>(), A, B,
                   REPLACE);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_Replace_bool_masked_result_ones)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<bool> M(LowerBool_3x4, false);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   M,
                   Second<double>(),
                   Times<double>(), A, B,
                   REPLACE);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_Replace_mask_stored_zero_result_ones)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(Lower_3x4, 0.);
    M.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   M,
                   Second<double>(),
                   Times<double>(), A, B,
                   REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_Mask_Accum_AB_Merge_Cones_Mlower)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);

    static std::vector<std::vector<double> > M_3x4 = {{1, 0, 0, 0},
                                                      {1, 1, 0, 0},
                                                      {1, 1, 1, 0}};
    Matrix<double> M(M_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   M,
                   Second<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// CompMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB)
{

    Matrix<double> A(A_sparse_3x3, 0.0);
    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > Not_A_sparse_3x3 =
        {{0,  0,  1},
         {1,  0,  1},
         {0,  1,  0}};
    Matrix<double> NotA(Not_A_sparse_3x3, 0.0);
    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
        {{12, 7,  1},
         {1, -5,  1},
         {7,  1,  9}};
    Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   complement(Ones), NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   complement(Empty), NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    kronecker(C,
                   complement(NotA), NoAccumulate(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, AFilled);

    C = Ones;
    kronecker(C,
                   complement(MNotLower), NoAccumulate(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   complement(MLower), NoAccumulate(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   complement(Ones), NoAccumulate(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   complement(Empty), NoAccumulate(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    kronecker(C,
                   complement(MNotLower), NoAccumulate(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    kronecker(C,
                   complement(MLower), NoAccumulate(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, MNotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_empty)
{
    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> Identity(Identity_3x3, 0.);
    Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    Matrix<bool> M(LowerBool_3x3, false);

    Matrix<double> mUpper(NotLower_3x3, 0.);

    // Merge
    C = Ones;
    kronecker(C,
                   complement(mUpper), NoAccumulate(),
                   Times<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    kronecker(C,
                   complement(mUpper), NoAccumulate(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    kronecker(C,
                   complement(Ones), NoAccumulate(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Empty;
    kronecker(C,
                   complement(Empty), NoAccumulate(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C,
                   complement(mUpper), NoAccumulate(),
                   Times<double>(), Empty, Ones, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   complement(mUpper), NoAccumulate(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    kronecker(C,
                   complement(Ones), NoAccumulate(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Empty;
    kronecker(C,
                   complement(Empty), NoAccumulate(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Ones);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_Merge_Cones_Mlower_stored_zero)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(NotLower_3x4, 0.);
    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   complement(M),
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> NotLower(NotLower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11,0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {0, 0, 1, 1},
                                                     {9, 0, 11,1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> NotLower(NotLower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{0, 0, 0, 0},
                                                    {0, 1, 0, 0},
                                                    {0, 4, 0, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{0, 1, 1, 1},
                                                     {0, 1, 1, 1},
                                                     {0, 4, 0, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> Lower(Lower_3x3, 0.);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 0, 7},
                                                    {0, 0, 0},
                                                    {0, 0, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   complement(Lower),
                   NoAccumulate(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 0, 7},
                                                     {1, 1, 0},
                                                     {1, 1, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   complement(Lower),
                   NoAccumulate(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_ACdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), C, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   NoAccumulate(),
                   Times<double>(), mat, C,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_NoAccum_AB_Replace_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);

    Matrix<double> result(Ones_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    Matrix<double> M(NotLower_4x4, 0.);

    kronecker(result,
                   complement(M),
                   NoAccumulate(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// CompMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB)
{
    Matrix<double> A(A_sparse_3x3, 0.0);
    Matrix<double> Identity(Identity_3x3, 0.0);

    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    Matrix<double> MLower(Lower_3x3, 0.);
    Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    static std::vector<std::vector<double> > Not_A_3x3 =
        {{0,  0,  1},
         {1,  0,  1},
         {0,  1,  0}};
    Matrix<double> NotA(Not_A_3x3, 0.0);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   complement(Ones), Plus<double>(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    //---
    static std::vector<std::vector<double> > ans =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    kronecker(C,
                   complement(NotA), Plus<double>(),
                   Times<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans, 0.));

    //---
    static std::vector<std::vector<double> > ans2 =
        {{2,  1,  1},
         {2,  2,  1},
         {2,  2,  2}};

    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans2, 0.));

    //---
    static std::vector<std::vector<double> > ans3 =
        {{1,  2,  2},
         {1,  1,  2},
         {1,  1,  1}};

    C = Ones;
    kronecker(C,
                   complement(MLower), Plus<double>(),
                   Times<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans3, 0.));

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    kronecker(C,
                   complement(Ones), Plus<double>(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);

    //---
    static std::vector<std::vector<double> > ans4 =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    kronecker(C,
                   complement(Empty), Plus<double>(),
                   Times<double>(), A, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans4, 0.));

    //---
    static std::vector<std::vector<double> > ans5 =
        {{2,  0,  0},
         {2,  2,  0},
         {2,  2,  2}};

    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans5, 0.));

    //---
    static std::vector<std::vector<double> > ans6 =
        {{0,  2,  2},
         {0,  0,  2},
         {0,  0,  0}};

    C = Ones;
    kronecker(C,
                   complement(MLower), Plus<double>(),
                   Times<double>(), Ones, Identity, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans6, 0.));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_ABM_empty)
{
    Matrix<double> Empty(3, 3);
    Matrix<double> Ones(Ones_3x3, 0.);
    Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    Matrix<bool> MNotLower(NotLowerBool_3x3, false);

    // Merge

    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    kronecker(C,
                   complement(Ones), Plus<double>(),
                   Times<double>(), Ones, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Empty, Ones, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    kronecker(C,
                   complement(MNotLower), Plus<double>(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    kronecker(C,
                   complement(Ones), Plus<double>(),
                   Times<double>(), Ones, Empty, REPLACE);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> MNotLower(NotLower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{2, 0, 0, 0},
                                                    {1, 1, 0, 0},
                                                    {10,1, 12,0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   complement(MNotLower),
                   Plus<double>(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{2, 1, 1, 1},
                                                     {1, 1, 1, 1},
                                                     {10,1,12,1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   complement(MNotLower),
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};

    Matrix<double> A(A_vals, 0.);
    Matrix<double> B(B_vals, 0.);
    Matrix<double> NotLower(NotLower_3x4, 0.);
    Matrix<double> Ones(Ones_3x4, 0.);
    Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {1, 2, 0, 0},
                                                    {1, 5, 1, 0}};
    Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), A, B, REPLACE);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {1, 2, 1, 1},
                                                     {1, 5, 1, 1}};
    Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  1,  1},
                                             {4, 10,  1,  1},
                                             {3, 11, 23,  1},
                                             {1,  7, 22, 26}};
    Matrix<double> answer(ans, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    Matrix<double> answer2(ans2, 0.);

    C = Ones;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_ACdup)
{

    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), C, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> Ones(Ones_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    Matrix<double> answer(ans, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    Matrix<double> answer2(ans2, 0.);

    C = mat;
    kronecker(C,
                   complement(NotLower),
                   Plus<double>(),
                   Times<double>(), mat, C,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);
    Matrix<double> NotLower(NotLower_4x4, 0.);
    Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    C = NotLower;
    kronecker(C,
                   complement(C),
                   Plus<double>(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    Matrix<double> answer2(ans2, 0.);

    C = NotLower;
    kronecker(C,
                   complement(C),
                   Plus<double>(),
                   Times<double>(), mat, mat,
                   REPLACE);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_Replace_Cones_Mnlower)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(NotLower_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   complement(M),
                   Second<double>(),
                   Times<double>(), A, B,
                   REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_Replace_Mstored_zero)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(NotLower_3x4, 0.);

    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   complement(M),
                   Second<double>(),
                   Times<double>(), A, B,
                   REPLACE);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_Merge)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(NotLower_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   complement(M),
                   Second<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_Merge_Mstored_zero)
{
    Matrix<double> A(A_dense_3x3, 0.);
    Matrix<double> B(B_dense_3x4, 0.);
    Matrix<double> M(NotLower_3x4, 0.);
    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    Matrix<double> answer(ans, 0.);

    kronecker(result,
                   complement(M),
                   Second<double>(),
                   Times<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_kronecker_CompMask_Accum_AB_Merge_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    Matrix<double> mat(m, 0.);

    Matrix<double> result(Ones_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    Matrix<double> answer(ans, 0.);

    Matrix<double> M(NotLower_4x4, 0.);

    kronecker(result,
                   complement(M),
                   NoAccumulate(),
                   Times<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
