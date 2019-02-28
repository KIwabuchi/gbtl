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
    static std::vector<std::vector<double> > mA_dense_3x3 =
            {{12, 7, 3},
             {4,  5, 6},
             {7,  8, 9}};

    static std::vector<std::vector<double> > mB_dense_3x4 =
            {{5, 8, 1, 2},
             {6, 7, 3, 0.},
             {4, 5, 9, 1}};

    static std::vector<std::vector<double> > mAnswer_dense =
            {{114, 160, 60,  27},
             {74,  97,  73,  14},
             {119, 157, 112, 23}};

    static std::vector<std::vector<double> > mA_sparse_3x3 =
            {{12.3, 7.5,  0},
             {0,    -5.2, 0},
             {7.0,  0,    9.0}};

    static std::vector<std::vector<double> > mA_sparse_1337zero_3x3 =
            {{12.3, 7.5,  1337},
             {1337, -5.2, 1337},
             {7.0,  1337, 9.0}};

    static std::vector<std::vector<double> > mB_sparse_3x4 =
            {{5.0, 8.5, 0,   -2.1},
             {0.0, -7,  3.8, 0.0},
             {4.0, 0,   0,   1.3}};

    static std::vector<std::vector<double> > mAnswer_sparse =
            {{61.5, 52.05, 28.5,   -25.83},
             {0.0,  36.4,  -19.76, 0.0},
             {71.0, 59.5,  0.0,    -3.0}};

    //static Matrix<double, DirectedMatrixTag> mAns(mAns_dense);
    GraphBLAS::IndexArrayType i_all3x4 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    GraphBLAS::IndexArrayType j_all3x4 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    static std::vector<std::vector<double> > mOnes_4x4 =
            {{1, 1, 1, 1},
             {1, 1, 1, 1},
             {1, 1, 1, 1},
             {1, 1, 1, 1}};

    static std::vector<std::vector<double> > mOnes_3x4 =
            {{1, 1, 1, 1},
             {1, 1, 1, 1},
             {1, 1, 1, 1}};

    static std::vector<std::vector<double> > mOnes_3x3 =
            {{1, 1, 1},
             {1, 1, 1},
             {1, 1, 1}};

    static std::vector<std::vector<double> > mIdentity_3x3 =
            {{1, 0, 0},
             {0, 1, 0},
             {0, 0, 1}};

    static std::vector<std::vector<double> > mLowerMask_3x4 =
            {{1, 0,    0,   0},
             {1, 0.5,  0,   0},
             {1, -1.0, 1.5, 0}};

    static std::vector<std::vector<bool> > mLowerBoolMask_3x4 =
            {{true, false, false, false},
             {true, true,  false, false},
             {true, true,  true,  false}};

}

//****************************************************************************
// checking result from graphblas test framework
BOOST_AUTO_TEST_CASE(graphblas_tf_mxm30)
{
    Matrix<double> A1(3,3), B1(3,3), C1(3,3);
    Matrix<bool> M1(3,3);

    //M1.setElement(3, 3, true);
    M1.setElement(0, 2, true);

    //A1.setElement(3, 3, 1.);
    A1.setElement(2, 1, 0.2215395618663762);

    //B1.setElement(3, 3, 1.);
    B1.setElement(0, 2, -1.796394937062196);

    //C1.setElement(3, 3, 1.);
    C1.setElement(0, 2, 0.7345667281066299);

    mxm(C1, GraphBLAS::complement(M1),
        GraphBLAS::Plus<double>(), ArithmeticSemiring<double>(),
        transpose(A1),transpose(B1));

    //GraphBLAS::print_matrix(std::cout, C1, "graphblas_tf Test 30");

    //===================
    C1.clear();
    C1.setElement(0, 2, 0.7345667281066299);

    mxm(C1, GraphBLAS::complement(M1),
        GraphBLAS::NoAccumulate(), ArithmeticSemiring<double>(),
        transpose(A1),transpose(B1), true);

    //GraphBLAS::print_matrix(std::cout, C1, "graphblas_tf Test 31");

    //===================
    C1.clear();
    C1.setElement(0, 2, 0.7345667281066299);

    mxm(C1, GraphBLAS::complement(M1),
        GraphBLAS::Plus<double>(), ArithmeticSemiring<double>(),
        transpose(A1),transpose(B1), true);

    //GraphBLAS::print_matrix(std::cout, C1, "graphblas_tf Test 32");
}

//****************************************************************************
// Original 9 tests from test_mxm
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_bad_dimensions)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    BOOST_CHECK_THROW(
        (mxm(result,
             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
             GraphBLAS::ArithmeticSemiring<double>(),
             mB, mA)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_reg)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_accum)
{
    // Build some matrices.
    IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double>       v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i, j, v);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer = {0, 0, 0,  1, 1, 1, 1,
                                          2, 2, 2, 2,  3, 3, 3};
    IndexArrayType j_answer = {0, 1, 2,  0, 1, 2, 3,
                                          0, 1, 2, 3,  1, 2, 3};
    std::vector<double> v_answer = {2, 3, 2,  3, 9, 10, 6,
                                    2, 10, 22, 21,  6, 21, 25};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(m3,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked2)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_not_all_one_masked)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {12, 4, 7, 7, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Original tests from test_sparse_mxm
//****************************************************************************

//****************************************************************************
// Tests without mask
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(3, 4);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x3(3, 3);

    // incompatible input matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mB, mA)),
        GraphBLAS::DimensionException);

    // incompatible output matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x3,
                        GraphBLAS::NoMask(),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(mAnswer_dense, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_reg_empty)
{
    std::vector<std::vector<double>> mA_vals = {{8, 1, 6},
                                                {0, 0, 0},
                                                {4, 9, 2}};

    std::vector<std::vector<double>> mB_vals = {{0, 0, 0, 1},
                                                {1, 0, 1, 1},
                                                {0, 0, 1, 1}};

    std::vector<std::vector<double>> answer_vals = {{1, 0, 7, 15},
                                                    {0, 0, 0, 0},
                                                    {9, 0, 11, 15}};

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_vals, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_vals, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(answer_vals, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_reg_sparse)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mC(3, 4);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_sparse_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_sparse_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(mAnswer_sparse, 0.);

    GraphBLAS::mxm(mC,
                   GraphBLAS::NoMask(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    for (GraphBLAS::IndexType ix = 0; ix < answer.nrows(); ++ix)
    {
        for (GraphBLAS::IndexType iy = 0; iy < answer.ncols(); ++iy)
        {
            BOOST_CHECK_EQUAL(mC.hasElement(ix, iy), answer.hasElement(ix, iy));
            if (mC.hasElement(ix, iy))
            {
                BOOST_CHECK_CLOSE(mC.extractElement(ix,iy),
                                  answer.extractElement(ix,iy), 0.0001);
            }
        }
    }
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_stored_zero_result)
{
    // Build some matrices.
    std::vector<std::vector<int> > A = {{1, 1, 0, 0},
                                        {1, 2, 2, 0},
                                        {0, 2, 3, 3},
                                        {0, 0, 3, 4}};
    std::vector<std::vector<int> > B = {{ 1,-2, 0,  0},
                                        {-1, 1, 0,  0},
                                        { 0, 0, 3, -4},
                                        { 0, 0,-3,  3}};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> mA(A, 0);
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> mB(B, 0);
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> result(4, 4);

    // use a different sentinel value so that stored zeros are preserved.
    int const NIL(666);
    std::vector<std::vector<int> > ans = {{  0,  -1, NIL, NIL},
                                          { -1,   0,   6,  -8},
                                          { -2,   2,   0,  -3},
                                          {NIL, NIL,  -3,   0}};
    GraphBLAS::Matrix<int, GraphBLAS::DirectedMatrixTag> answer(ans, NIL);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<int>(),
                   GraphBLAS::ArithmeticSemiring<int>(), mA, mB);
    BOOST_CHECK_EQUAL(result, answer);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(4, 4);

    std::vector<std::vector<double> > ans = {{2,  3,  2,  0},
                                             {3,  9, 10,  6},
                                             {2, 10, 22, 21},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose2)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 4);

    std::vector<std::vector<double> > ans = {{112, 159,  87, 31},
                                             { 97, 131,  94, 22},
                                             { 87, 111, 102, 15}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose2)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans = {{119,  87, 79},
                                             { 66,  80, 62},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose2)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans =  {{99,  97, 87},
                                              {83, 100, 81},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose2)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(3, 3);

    std::vector<std::vector<double> > ans = {{12, 4, 7},
                                             { 7, 5, 8},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxm(result,
                   GraphBLAS::NoMask(),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with REPLACE
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        mMask,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                        true)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    // NOTE: The mask is true for any non-zero.
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mLowerMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_reg_bool)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<bool, GraphBLAS::DirectedMatrixTag> mMask(mLowerBoolMask_3x4, false);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);

    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_replace_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a mask with MERGE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        mMask,
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 1, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
//                   GraphBLAS::Second<double>(),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{1, 0, 0, 0},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 1, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxm(result,
                   mMask,
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask with REPLACE semantics
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::complement(mMask),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                        true)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   0,   0,  0},
                                             { 74,  97,   0,  0},
                                             {119, 157, 112,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB,
                   true);
    BOOST_CHECK_EQUAL(result.nvals(), 6);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  0,  0,  0},
                                             {3,  9,  0,  0},
                                             {2, 10, 22,  0},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   0,   0,  0},
                                             { 97, 131,   0,  0},
                                             { 87, 111, 102,  0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   0,  0},
                                             { 66,  80,  0},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   0,  0},
                                              {83, 100,  0},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB),
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_replace_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 0, 0},
                                             { 7, 5, 0},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB,
                   true);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Tests using a complemented mask (with merge semantics)
//****************************************************************************

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_bad_dimensions)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.); // 3x3
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.); // 3x4
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result3x4(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{1, 0, 0},
                                                          {1, 1, 0},
                                                          {1, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    // incompatible mask matrix dimensions
    BOOST_CHECK_THROW(
        (GraphBLAS::mxm(result3x4,
                        GraphBLAS::complement(mMask),
                        GraphBLAS::Second<double>(),
                        GraphBLAS::ArithmeticSemiring<double>(), mA, mB)),
        GraphBLAS::DimensionException);
}
//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_reg)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 6);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_reg_stored_zero)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);
    mMask.setElement(0, 0, 0.);
    BOOST_CHECK_EQUAL(mMask.nvals(), 7);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    std::vector<std::vector<double> > ans = {{114,   1,   1,  1},
                                             { 74,  97,   1,  1},
                                             {119, 157, 112,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, mB);
    BOOST_CHECK_EQUAL(result.nvals(), 12);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_duplicate_input)
{
    // Build some matrices.
    std::vector<std::vector<double> > m = {{1, 1, 0, 0},
                                           {1, 2, 2, 0},
                                           {0, 2, 3, 3},
                                           {0, 0, 3, 4}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mat(m, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_4x4, 0.);

    std::vector<std::vector<double> > ans = {{2,  1,  1,  1},
                                             {3,  9,  1,  1},
                                             {2, 10, 22,  1},
                                             {0,  6, 21, 25}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    static std::vector<std::vector<double> > mMask_4x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_4x4, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::NoAccumulate(),
                   GraphBLAS::ArithmeticSemiring<double>(), mat, mat);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_a_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mB_dense_3x4, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x4, 0.);

    static std::vector<std::vector<double> > mMask_3x4 = {{0, 1, 1, 1},
                                                          {0, 0, 1, 1},
                                                          {0, 0, 0, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x4, 0.);

    std::vector<std::vector<double> > ans = {{112,   1,   1,  1},
                                             { 97, 131,   1,  1},
                                             { 87, 111, 102,  1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);

    std::vector<std::vector<double> > B = {{5, 8, 1},
                                           {2, 6, 7},
                                           {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{119,   1,  1},
                                             { 66,  80,  1},
                                             {108, 125, 98}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_a_and_b_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    std::vector<std::vector<double> > B =  {{5, 8, 1},
                                            {2, 6, 7},
                                            {3, 4, 5}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(B, 0.);

    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans =  {{99,   1,  1},
                                              {83, 100,  1},
                                              {72, 105, 78}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_scmp_masked_a_equals_c_transpose)
{
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mA(mA_dense_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mB(mIdentity_3x3, 0.);
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> result(mOnes_3x3, 0.);

    static std::vector<std::vector<double> > mMask_3x3 = {{0, 1, 1},
                                                          {0, 0, 1},
                                                          {0, 0, 0}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> mMask(mMask_3x3, 0.);

    std::vector<std::vector<double> > ans = {{12, 1, 1},
                                             { 7, 5, 1},
                                             { 3, 6, 9}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> answer(ans, 0.);

    //auto answer = GraphBLAS::transpose(mA);

    GraphBLAS::mxm(result,
                   GraphBLAS::complement(mMask),
                   GraphBLAS::Second<double>(),
                   GraphBLAS::ArithmeticSemiring<double>(), transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_bad_dimensions2)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {114, 160, 60, 27, 74, 97,
                                    73, 14, 119, 157, 112, 23};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    BOOST_CHECK_THROW(
        (mxm(result,
             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
             GraphBLAS::ArithmeticSemiring<double>(),
             mB, mA)),
        DimensionException);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_reg2)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(mxm_accum2)
{
    // Build some matrices.
    IndexArrayType i = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    IndexArrayType j = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<double>       v = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
    Matrix<double, DirectedMatrixTag> mat(4, 4);
    mat.build(i, j, v);

    Matrix<double, DirectedMatrixTag> m3(4, 4);

    IndexArrayType i_answer = {0, 0, 0,  1, 1, 1, 1,
                                          2, 2, 2, 2,  3, 3, 3};
    IndexArrayType j_answer = {0, 1, 2,  0, 1, 2, 3,
                                          0, 1, 2, 3,  1, 2, 3};
    std::vector<double> v_answer = {2, 3, 2,  3, 9, 10, 6,
                                    2, 10, 22, 21,  6, 21, 25};
    Matrix<double, DirectedMatrixTag> answer(4, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(m3,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mat, mat);

    BOOST_CHECK_EQUAL(m3, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_masked)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_mxm_not_all_one_masked2)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

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
        mA, mB);

    BOOST_CHECK_EQUAL(result, answer);
}


//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_transpose3)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5, 9, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 4);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 4);

    IndexArrayType i_answer = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> v_answer = {112, 159, 87, 31, 97, 131,
                                    94, 22, 87, 111, 102, 15};
    Matrix<double, DirectedMatrixTag> answer(3, 4);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_b_transpose3)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {119, 87, 79, 66, 80, 62, 108, 125, 98};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        mA, transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_and_b_transpose3)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {5, 8, 1, 2, 6, 7, 3, 4, 5};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {99, 97, 87, 83, 100, 81, 72, 105, 78};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), transpose(mB));

    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
BOOST_AUTO_TEST_CASE(test_a_equals_c_transpose3)
{
    IndexArrayType i_mA    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mA    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mA = {12, 7, 3, 4, 5, 6, 7, 8, 9};
    Matrix<double, DirectedMatrixTag> mA(3, 3);
    mA.build(i_mA, j_mA, v_mA);

    IndexArrayType i_mB    = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_mB    = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_mB = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Matrix<double, DirectedMatrixTag> mB(3, 3);
    mB.build(i_mB, j_mB, v_mB);

    Matrix<double, DirectedMatrixTag> result(3, 3);

    IndexArrayType i_answer = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    IndexArrayType j_answer = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    std::vector<double> v_answer = {12, 4, 7, 7, 5, 8, 3, 6, 9};
    Matrix<double, DirectedMatrixTag> answer(3, 3);
    answer.build(i_answer, j_answer, v_answer);

    mxm(result,
        GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
        GraphBLAS::ArithmeticSemiring<double>(),
        transpose(mA), mB);

    BOOST_CHECK_EQUAL(result, answer);
}

BOOST_AUTO_TEST_SUITE_END()
