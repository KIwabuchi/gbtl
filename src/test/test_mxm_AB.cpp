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

//#define GRAPHBLAS_LOGGING_LEVEL 2

#include <graphblas/graphblas.hpp>

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mxm_test_suite

#include <boost/test/included/unit_test.hpp>

using namespace GraphBLAS;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

//****************************************************************************

namespace
{
    static std::vector<std::vector<double> > A_dense_3x3 =
    {{12, 7, 3},
     {4,  5, 6},
     {7,  8, 9}};

    static std::vector<std::vector<double> > AT_dense_3x3 =
    {{12, 4, 7},
     {7,  5, 8},
     {3,  6, 9}};

    static std::vector<std::vector<double> > B_dense_3x4 =
    {{5, 8, 1, 2},
     {6, 7, 3, 0.},
     {4, 5, 9, 1}};

    static std::vector<std::vector<double> > BT_dense_3x4 =
    {{5, 6, 4},
     {8, 7, 5},
     {1, 3, 9},
     {2, 0, 1}};

    static std::vector<std::vector<double> > Answer_dense =
    {{114, 160, 60,  27},
     {74,  97,  73,  14},
     {119, 157, 112, 23}};

    static std::vector<std::vector<double> > Answer_plus1_dense =
    {{115, 161, 61,  28},
     {75,  98,  74,  15},
     {120, 158, 113, 24}};

    static std::vector<std::vector<double> > A_sparse_3x3 =
    {{12, 7,  0},
     {0, -5,  0},
     {7,  0,  9}};

    static std::vector<std::vector<double> > AT_sparse_3x3 =
    {{12, 7,  0},
     {0, -5,  0},
     {7,  0,  9}};

    static std::vector<std::vector<double> > B_sparse_3x4 =
    {{5., 8.,  0, -2.},
     {0., -7,  3., 0.},
     {4., 0,   0,  1.}};

    static std::vector<std::vector<double> > BT_sparse_3x4 =
    {{5.,  0., 4},
     {8., -7,  0.},
     {0.,  3,  0.},
     {-2., 0,  1}};

    // A_sparse_3x3 * A_sparse_3x3
    static std::vector<std::vector<double> > AA_answer_sparse =
    {{144.,  49., 0},
     {0.0,   25., 0},
     {147.,  49., 81.}};

    // A_sparse_3x3 * B_sparse_3x4
    static std::vector<std::vector<double> > Answer_sparse =
    {{60,   47., 21,  -24},
     {0.0,  35.,-15,  0.0},
     {71.0, 56,  0.0, -5.0}};

    //static Matrix<double, DirectedMatrixTag> Ans(Ans_dense);
    GraphBLAS::IndexArrayType i_all3x4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_all3x4 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

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

}

//****************************************************************************
// API error tests
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> A(A_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> B(B_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(3, 4);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x3(3, 3);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> ones3x4(Ones_3x4, 0.);

    static std::vector<std::vector<double> > M_3x3 = {{1, 0, 0},
                                                      {1, 1, 0},
                                                      {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> M(M_3x3, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    // NoMask_NoAccum_AB

    // ncols(A) != nrows(B)
    BOOST_CHECK_THROW(
        (mxm(result3x4,
             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
             GraphBLAS::ArithmeticSemiring<double>(),
             B, A)),
        DimensionException);

    // dim(C) != dim(A*B)
    BOOST_CHECK_THROW(
        (mxm(result3x3,
             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
             GraphBLAS::ArithmeticSemiring<double>(),
             A, B)),
        DimensionException);

    // NoMask_Accum_AB

    // incompatible input matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), B, A)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x3,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B)),
        GraphBLAS::DimensionException);

    // Mask_NoAccum

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(ones3x4,
                        M,
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B,
                        true)),
        GraphBLAS::DimensionException);

    // Mask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(ones3x4,
                        M,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B, true)),
        GraphBLAS::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(ones3x4,
                        M,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B)),
        GraphBLAS::DimensionException);

    // CompMask_NoAccum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(ones3x4,
                        GraphBLAS::complement(M),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B, true)),
        GraphBLAS::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::complement(M),
                        GraphBLAS::NoAccumulate(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B)),
        GraphBLAS::DimensionException);

    // CompMask_Accum (replace and merge)

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(ones3x4,
                        GraphBLAS::complement(M),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B, true)),
        GraphBLAS::DimensionException);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::complement(M),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), A, B)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
// NoMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB)
{
    GraphBLAS::Matrix<double> C(3, 4);
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_sparse_3x4, 0.);

    GraphBLAS::Matrix<double> answer(Answer_sparse, 0.);

    GraphBLAS::mxm(C,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    for (GraphBLAS::IndexType ix = 0; ix < answer.nrows(); ++ix)
    {
        for (GraphBLAS::IndexType iy = 0; iy < answer.ncols(); ++iy)
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
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_empty)
{
    GraphBLAS::Matrix<double> Zero(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> mD(Ones_3x3, 0.);

    GraphBLAS::mxm(C,
                   NoMask(), NoAccumulate(),
                   ArithmeticSemiring<double>(), Zero, Ones);
    BOOST_CHECK_EQUAL(C, Zero);

    GraphBLAS::mxm(mD,
                   NoMask(), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Zero);
    BOOST_CHECK_EQUAL(mD, Zero);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_dense)
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

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        A, B);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    std::vector<std::vector<double>> answer_vals = {{1, 0, 7, 15},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11, 15}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> result(3, 4);
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};

    std::vector<std::vector<double>> answer_vals = {{0, 8, 0, 8},
                                                    {0, 1, 0, 1},
                                                    {0, 4, 0, 4}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> result(3, 4);
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    IndexArrayType      i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType      j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double> v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i, j, v);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType      i_answer = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
    IndexArrayType      j_answer = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3};
    std::vector<double> v_answer = {2, 3, 2, 3, 9,10, 6, 2,10,22,21, 6,21,25};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(m3,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_ACdup)
{
    GraphBLAS::Matrix<double> C(A_sparse_3x3, 0.);
    GraphBLAS::Matrix<double> B(A_sparse_3x3, 0.);

    GraphBLAS::Matrix<double> answer(AA_answer_sparse, 0.);

    GraphBLAS::mxm(C,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, B);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_NoAccum_AB_BCdup)
{
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.);
    GraphBLAS::Matrix<double> C(B_sparse_3x4, 0.);

    GraphBLAS::Matrix<double> answer(Answer_sparse, 0.);

    GraphBLAS::mxm(C,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
// NoMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double> result(3, 4);
    GraphBLAS::Matrix<double> answer(Answer_dense, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_empty)
{
    GraphBLAS::Matrix<double> Zero(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> mD(Ones_3x3, 0.);

    GraphBLAS::mxm(C,
                   NoMask(), Plus<double>(),
                   ArithmeticSemiring<double>(), Zero, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    GraphBLAS::mxm(mD,
                   NoMask(), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Zero);
    BOOST_CHECK_EQUAL(mD, Ones);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A_mat = {{1, 1, 0, 0},
                                            {1, 2, 2, 0},
                                            {0, 2, 3, 3},
                                            {0, 0, 3, 4}};
    std::vector<std::vector<int> > B_mat = {{ 1,-2, 0,  0},
                                            {-1, 1, 0,  0},
                                            { 0, 0, 3, -4},
                                            { 0, 0,-3,  3}};
    GraphBLAS::Matrix<int> A(A_mat, 0);
    GraphBLAS::Matrix<int> B(B_mat, 0);
    GraphBLAS::Matrix<int> result(4, 4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<std::vector<int> > ans = {{  0,  -1, NIL, NIL},
                                          { -1,   0,   6,  -8},
                                          { -2,   2,   0,  -3},
                                          {NIL, NIL,  -3,   0}};
    GraphBLAS::Matrix<int> answer(ans, NIL);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<int>(),
                   GraphBLAS::ArithmeticSemiring<int>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_ABdup_Cempty)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);

    GraphBLAS::Matrix<double> result(4, 4);

    std::vector<std::vector<double> > ans = {{2,  3,  2,  0},
                                             {3,  9, 10,  6},
                                             {2, 10, 22, 21},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_emptyRowA_emptyColB)
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

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_emptyColA_emptyRowB)
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

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_ABdup)
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

    mxm(m3,
        GraphBLAS::NoMask(), GraphBLAS::Plus<double>(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_ACdup)
{
    GraphBLAS::Matrix<double> C(A_sparse_3x3, 0.);
    GraphBLAS::Matrix<double> B(A_sparse_3x3, 0.);

    // A_sparse_3x3 * A_sparse_3x3 + A_sparse_3x3
    static std::vector<std::vector<double> > ans =
        {{156.,  56., 0},
         {0.0,   20., 0},
         {154.,  49., 90.}};
    Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(C,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, B);

    BOOST_CHECK_EQUAL(C, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_NoMask_Accum_AB_BCdup)
{
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.);
    GraphBLAS::Matrix<double> C(A_sparse_3x3, 0.);

    // A_sparse_3x3 * A_sparse_3x3 + A_sparse_3x3
    static std::vector<std::vector<double> > ans =
        {{156.,  56., 0},
         {0.0,   20., 0},
         {154.,  49., 90.}};
    Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(C,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, C);

    BOOST_CHECK_EQUAL(C, answer);
}

// ****************************************************************************
// Mask_NoAccum
// ****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB)
{
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.0);
    GraphBLAS::Matrix<double> Identity(Identity_3x3, 0.0);

    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    GraphBLAS::Matrix<double> MLower(Lower_3x3, 0.);
    GraphBLAS::Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    GraphBLAS::Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   Ones, NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    GraphBLAS::mxm(C,
                   A, NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, AFilled);

    C = Ones;
    GraphBLAS::mxm(C,
                   MLower, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   MNotLower, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   Ones, NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    GraphBLAS::mxm(C,
                   MLower, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    GraphBLAS::mxm(C,
                   MNotLower, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, MNotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_ABM_empty)
{
    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    GraphBLAS::Matrix<bool> M(LowerBool_3x3, false);

    GraphBLAS::Matrix<double> mUpper(NotLower_3x3, 0.);

    // Merge
    C = Ones;
    GraphBLAS::mxm(C,
                   M, NoAccumulate(),
                   ArithmeticSemiring<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    GraphBLAS::mxm(C,
                   M, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    GraphBLAS::mxm(C,
                   M, NoAccumulate(),
                   ArithmeticSemiring<double>(), Empty, Ones, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   M, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);

}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Merge_full_mask)
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

    mxm(result,
        mask, GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        A, B);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_mask_not_full)
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

    mxm(result,
        mask, GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        A, B);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_Merge_Cones_Mlower_stored_zero)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(Lower_3x4, 0.);
    M.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   M,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11,0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {0, 0, 1, 1},
                                                     {9, 0, 11,1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{0, 0, 0, 0},
                                                    {0, 1, 0, 0},
                                                    {0, 4, 0, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{0, 1, 1, 1},
                                                     {0, 1, 1, 1},
                                                     {0, 4, 0, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_3x3, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 0, 7},
                                                    {0, 0, 0},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   NotLower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 0, 7},
                                                     {1, 1, 0},
                                                     {1, 1, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   NotLower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_ACdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_NoAccum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Lower;
    GraphBLAS::mxm(C,
                   C,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Lower;
    GraphBLAS::mxm(C,
                   C,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
// Mask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB)
{
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.0);
    GraphBLAS::Matrix<double> Identity(Identity_3x3, 0.0);

    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    GraphBLAS::Matrix<double> MLower(Lower_3x3, 0.);
    GraphBLAS::Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    GraphBLAS::Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    //---
    static std::vector<std::vector<double> > ans =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    GraphBLAS::mxm(C,
                   A, Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans, 0.));

    //---
    static std::vector<std::vector<double> > ans2 =
        {{2,  1,  1},
         {2,  2,  1},
         {2,  2,  2}};

    C = Ones;
    GraphBLAS::mxm(C,
                   MLower, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans2, 0.));

    //---
    static std::vector<std::vector<double> > ans3 =
        {{1,  2,  2},
         {1,  1,  2},
         {1,  1,  1}};

    C = Ones;
    GraphBLAS::mxm(C,
                   MNotLower, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans3, 0.));

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Empty);

    //---
    static std::vector<std::vector<double> > ans4 =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    GraphBLAS::mxm(C,
                   Ones, Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans4, 0.));

    //---
    static std::vector<std::vector<double> > ans5 =
        {{2,  0,  0},
         {2,  2,  0},
         {2,  2,  2}};

    C = Ones;
    GraphBLAS::mxm(C,
                   MLower, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans5, 0.));

    //---
    static std::vector<std::vector<double> > ans6 =
        {{0,  2,  2},
         {0,  0,  2},
         {0,  0,  0}};

    C = Ones;
    GraphBLAS::mxm(C,
                   MNotLower, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans6, 0.));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_ABMempty)
{
    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    GraphBLAS::Matrix<bool> M(LowerBool_3x3, false);

    // Merge

    C = Ones;
    GraphBLAS::mxm(C,
                   M, Plus<double>(),
                   ArithmeticSemiring<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   M, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    GraphBLAS::mxm(C,
                   M, Plus<double>(),
                   ArithmeticSemiring<double>(), Empty, Ones, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    GraphBLAS::mxm(C,
                   M, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    GraphBLAS::mxm(C,
                   Empty, Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{2, 0, 0, 0},
                                                    {1, 1, 0, 0},
                                                    {10,1, 12,0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{2, 1, 1, 1},
                                                     {1, 1, 1, 1},
                                                     {10,1,12,1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {1, 2, 0, 0},
                                                    {1, 5, 1, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {1, 2, 1, 1},
                                                     {1, 5, 1, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_3x3, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 1, 8},
                                                    {0, 0, 1},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   NotLower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 8},
                                                     {1, 1, 1},
                                                     {1, 1, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   NotLower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  1,  1},
                                             {4, 10,  1,  1},
                                             {3, 11, 23,  1},
                                             {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_ACdup)
{

    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   Lower,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  0,  0,  0},
                                             {4, 10,  0,  0},
                                             {3, 11, 23,  0},
                                             {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Lower;
    GraphBLAS::mxm(C,
                   C,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Lower;
    GraphBLAS::mxm(C,
                   C,
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_lower_mask_result_ones)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(LowerMask_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   M,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B,
                   true);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_bool_masked_result_ones)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<bool> M(LowerBool_3x4, false);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   M,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B,
                   true);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Replace_mask_stored_zero_result_ones)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(Lower_3x4, 0.);
    M.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   M,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_Mask_Accum_AB_Merge_Cones_Mlower)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);

    static std::vector<std::vector<double> > M_3x4 = {{1, 0, 0, 0},
                                                      {1, 1, 0, 0},
                                                      {1, 1, 1, 0}};
    GraphBLAS::Matrix<double> M(M_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   M,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// CompMask_NoAccum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB)
{

    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.0);
    GraphBLAS::Matrix<double> Identity(Identity_3x3, 0.0);

    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    GraphBLAS::Matrix<double> MLower(Lower_3x3, 0.);
    GraphBLAS::Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > Not_A_sparse_3x3 =
        {{0,  0,  1},
         {1,  0,  1},
         {0,  1,  0}};
    GraphBLAS::Matrix<double> NotA(Not_A_sparse_3x3, 0.0);
    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
        {{12, 7,  1},
         {1, -5,  1},
         {7,  1,  9}};
    GraphBLAS::Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Empty), NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotA), NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, AFilled);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MLower), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Empty), NoAccumulate(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, A);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, MLower);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MLower), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, MNotLower);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_empty)
{
    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> Identity(Identity_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    GraphBLAS::Matrix<bool> M(LowerBool_3x3, false);

    GraphBLAS::Matrix<double> mUpper(NotLower_3x3, 0.);

    // Merge
    C = Ones;
    GraphBLAS::mxm(C,
                   complement(mUpper), NoAccumulate(),
                   ArithmeticSemiring<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(mUpper), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, mUpper);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Empty;
    GraphBLAS::mxm(C,
                   complement(Empty), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    GraphBLAS::mxm(C,
                   complement(mUpper), NoAccumulate(),
                   ArithmeticSemiring<double>(), Empty, Ones, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(mUpper), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);

    C = Empty;
    GraphBLAS::mxm(C,
                   complement(Empty), NoAccumulate(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, Ones);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_Merge_Cones_Mlower_stored_zero)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(NotLower_3x4, 0.);
    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   complement(M),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11,0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {0, 0, 1, 1},
                                                     {9, 0, 11,1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{0, 0, 0, 0},
                                                    {0, 1, 0, 0},
                                                    {0, 4, 0, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{0, 1, 1, 1},
                                                     {0, 1, 1, 1},
                                                     {0, 4, 0, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_emptyRowM)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0},
                                               {1, 0, 1},
                                               {0, 0, 1}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> Lower(Lower_3x3, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // Replace
    std::vector<std::vector<double>> answer_vals = {{0, 0, 7},
                                                    {0, 0, 0},
                                                    {0, 0, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Lower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 0, 7},
                                                     {1, 1, 0},
                                                     {1, 1, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Lower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_ACdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  0,  0},
                                             {3,  9,  2,  0},
                                             {2, 10, 22,  3},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_NoAccum_AB_Replace_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);

    GraphBLAS::Matrix<double> result(Ones_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::Matrix<double> M(NotLower_4x4, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// CompMask_Accum
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB)
{
    GraphBLAS::Matrix<double> A(A_sparse_3x3, 0.0);
    GraphBLAS::Matrix<double> Identity(Identity_3x3, 0.0);

    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    GraphBLAS::Matrix<double> MLower(Lower_3x3, 0.);
    GraphBLAS::Matrix<double> MNotLower(NotLower_3x3, 0.);

    static std::vector<std::vector<double> > A_sparse_fill_in_3x3 =
    {{12, 7,  1},
     {1, -5,  1},
     {7,  1,  9}};
    GraphBLAS::Matrix<double> AFilled(A_sparse_fill_in_3x3, 0.0);


    static std::vector<std::vector<double> > Not_A_3x3 =
        {{0,  0,  1},
         {1,  0,  1},
         {0,  1,  0}};
    GraphBLAS::Matrix<double> NotA(Not_A_3x3, 0.0);

    // Merge
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Ones);

    //---
    static std::vector<std::vector<double> > ans =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotA), Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans, 0.));

    //---
    static std::vector<std::vector<double> > ans2 =
        {{2,  1,  1},
         {2,  2,  1},
         {2,  2,  2}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans2, 0.));

    //---
    static std::vector<std::vector<double> > ans3 =
        {{1,  2,  2},
         {1,  1,  2},
         {1,  1,  1}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans3, 0.));

    // Replace
    // Mempty vs Mfull vs Mlower

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Empty);

    //---
    static std::vector<std::vector<double> > ans4 =
        {{13, 8,  1},
         {1, -4,  1},
         {8,  1, 10}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Empty), Plus<double>(),
                   ArithmeticSemiring<double>(), A, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans4, 0.));

    //---
    static std::vector<std::vector<double> > ans5 =
        {{2,  0,  0},
         {2,  2,  0},
         {2,  2,  2}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans5, 0.));

    //---
    static std::vector<std::vector<double> > ans6 =
        {{0,  2,  2},
         {0,  0,  2},
         {0,  0,  0}};

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Identity, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(ans6, 0.));
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_ABM_empty)
{
    GraphBLAS::Matrix<double> Empty(3, 3);
    GraphBLAS::Matrix<double> Ones(Ones_3x3, 0.);
    GraphBLAS::Matrix<double> C(3,3);

    // NOTE: The mask is true for any non-zero.
    GraphBLAS::Matrix<bool> MNotLower(NotLowerBool_3x3, false);

    // Merge

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Empty, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty);
    BOOST_CHECK_EQUAL(C, Ones);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Ones);
    BOOST_CHECK_EQUAL(C, Ones);

    // Replace
    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Empty, Ones, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Matrix<double>(Lower_3x3, 0.));

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(Ones), Plus<double>(),
                   ArithmeticSemiring<double>(), Ones, Empty, true);
    BOOST_CHECK_EQUAL(C, Empty);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_emptyRowA_emptyColB)
{
    std::vector<std::vector<double>> A_vals = {{8, 1, 6},
                                               {0, 0, 0},
                                               {4, 9, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 0, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 1, 1}};

    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> MNotLower(NotLower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{2, 0, 0, 0},
                                                    {1, 1, 0, 0},
                                                    {10,1, 12,0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{2, 1, 1, 1},
                                                     {1, 1, 1, 1},
                                                     {10,1,12,1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(MNotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_emptyColA_emptyRowB)
{
    std::vector<std::vector<double>> A_vals = {{8, 0, 6},
                                               {1, 0, 9},
                                               {4, 0, 2}};

    std::vector<std::vector<double>> B_vals = {{0, 1, 0, 1},
                                               {1, 0, 1, 1},
                                               {0, 0, 0, 0}};


    GraphBLAS::Matrix<double> A(A_vals, 0.);
    GraphBLAS::Matrix<double> B(B_vals, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_3x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_3x4, 0.);
    GraphBLAS::Matrix<double> C(3, 4);

    // REPLACE
    std::vector<std::vector<double>> answer_vals = {{1, 0, 0, 0},
                                                    {1, 2, 0, 0},
                                                    {1, 5, 1, 0}};
    GraphBLAS::Matrix<double> answer(answer_vals, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B, true);
    BOOST_CHECK_EQUAL(C, answer);

    // Merge
    std::vector<std::vector<double>> answer_vals2 = {{1, 1, 1, 1},
                                                     {1, 2, 1, 1},
                                                     {1, 5, 1, 1}};
    GraphBLAS::Matrix<double> answer2(answer_vals2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  1,  1},
                                             {4, 10,  1,  1},
                                             {3, 11, 23,  1},
                                             {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 10,  0,  0},
                                              {3, 11, 23,  0},
                                              {1,  7, 22, 26}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = Ones;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_ACdup)
{

    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), C, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_BCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> Ones(Ones_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{3,  1,  0,  0},
                                             {4, 11,  2,  0},
                                             {2, 12, 25,  3},
                                             {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C);

    BOOST_CHECK_EQUAL(C, answer);

    // Double check previous operation (without duplicating)
    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{3,  0,  0,  0},
                                              {4, 11,  0,  0},
                                              {2, 12, 25,  0},
                                              {0,  6, 24, 29}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = mat;
    GraphBLAS::mxm(C,
                   complement(NotLower),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, C,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_MCdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);
    GraphBLAS::Matrix<double> NotLower(NotLower_4x4, 0.);
    GraphBLAS::Matrix<double> C(4,4);

    // Merge
    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    C = NotLower;
    GraphBLAS::mxm(C,
                   complement(C),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(C, answer);

    // Replace
    std::vector<std::vector<double> > ans2 = {{2,  0,  0,  0},
                                              {3,  9,  0,  0},
                                              {2, 10, 22,  0},
                                              {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer2(ans2, 0.);

    C = NotLower;
    GraphBLAS::mxm(C,
                   complement(C),
                   GraphBLAS::Plus<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(C, answer2);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Replace_Cones_Mnlower)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(NotLower_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Replace_Mstored_zero)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(NotLower_3x4, 0.);

    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(NotLower_3x4, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 6);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge_Mstored_zero)
{
    GraphBLAS::Matrix<double> A(A_dense_3x3, 0.);
    GraphBLAS::Matrix<double> B(B_dense_3x4, 0.);
    GraphBLAS::Matrix<double> M(NotLower_3x4, 0.);
    M.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(M.nvals(), 7);

    GraphBLAS::Matrix<double> result(Ones_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), A, B);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_CompMask_Accum_AB_Merge_ABdup)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double> mat(m, 0.);

    GraphBLAS::Matrix<double> result(Ones_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double> answer(ans, 0.);

    GraphBLAS::Matrix<double> M(NotLower_4x4, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(M),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
