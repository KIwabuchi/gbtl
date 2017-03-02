/*
 * Copyright (c) 2017 Carnegie Mellon University.
 * All Rights Reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY AND THE TRUSTEES OF INDIANA UNIVERSITY EXPRESSLY DISCLAIM
 * TO THE FULLEST EXTENT PERMITTED BY LAW ALL EXPRESS, IMPLIED, AND STATUTORY
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *
 * This Program is distributed under a BSD license.  Please see LICENSE file or
 * permission@sei.cmu.edu for more information.  DM-0002659
 */

#include <iostream>
#include <graphblas/graphblas.hpp>

using namespace GraphBLAS;

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE mxm_suite

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(mxm_suite)

//****************************************************************************
// Matrix multiply
BOOST_AUTO_TEST_CASE(mxm_reg)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    
    std::vector<std::vector<double>> mat2 = {{0, 1, 1},
                                             {1, 0, 1},
                                             {1, 1, 0}};
    
    std::vector<std::vector<double>> mat3 = {{7, 14, 9},
                                             {12, 10, 8},
                                             {11, 6, 13}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);
    std::cout << "\nMatrices built";
    GraphBLAS::backend::mxm(result,
                            GraphBLAS::Second<double>(),                // accum
                            GraphBLAS::ArithmeticSemiring<double>(),    // semiring
                            m1,
                            m2);
    std::cout << "\nmxm computed";
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply, empty rows and columns
BOOST_AUTO_TEST_CASE(mxm_reg_empty)
{
    std::vector<std::vector<double>> mat1 = {{8, 1, 6},
                                             {3, 5, 7},
                                             {4, 9, 2}};
    
    std::vector<std::vector<double>> mat2 = {{0, 0, 0},
                                             {1, 0, 1},
                                             {0, 0, 1}};
    
    std::vector<std::vector<double>> mat3 = {{1, 0, 7},
                                             {5, 0, 12},
                                             {9, 0, 11}};
    
    GraphBLAS::LilSparseMatrix<double> m1(mat1, 0);
    GraphBLAS::LilSparseMatrix<double> m2(mat2, 0);
    GraphBLAS::LilSparseMatrix<double> answer(mat3, 0);
    GraphBLAS::IndexType M = 3;
    GraphBLAS::IndexType N = 3;
    GraphBLAS::LilSparseMatrix<double> result(M, N);
    GraphBLAS::backend::mxm(result,
                            GraphBLAS::Second<double>(),
                            GraphBLAS::ArithmeticSemiring<double>(),
                            m1,
                            m2);
    BOOST_CHECK_EQUAL(result, answer);
}

//****************************************************************************
// Matrix multiply, with accum


BOOST_AUTO_TEST_SUITE_END()
