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

#pragma once

#include <cstddef>
#include <graphblas/platforms/optimized_sequential/LilSparseMatrix.hpp>

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //********************************************************************
        template<typename ScalarT, typename... TagsT>
        class Matrix : public LilSparseMatrix<ScalarT>
        {
        public:
            typedef ScalarT ScalarType;

            LilSparseMatrix<ScalarT> &getBase() {return *this;}
            LilSparseMatrix<ScalarT> const &getBase() const {return *this;}

            // construct an empty matrix of fixed dimensions
            Matrix(IndexType   num_rows,
                   IndexType   num_cols)
                : LilSparseMatrix<ScalarT>(num_rows, num_cols)
            {
            }

            // copy construct
            Matrix(Matrix const &rhs)
                : LilSparseMatrix<ScalarT>(rhs)
            {
            }

            // construct a dense matrix from dense data.
            Matrix(std::vector<std::vector<ScalarT> > const &values)
                : LilSparseMatrix<ScalarT>(values)
            {
            }

            // construct a sparse matrix from dense data and a zero val.
            Matrix(std::vector<std::vector<ScalarT> > const &values,
                   ScalarT                                   zero)
                : LilSparseMatrix<ScalarT>(values, zero)
            {
            }

            ~Matrix() {}  // virtual?

            // necessary?
            bool operator==(Matrix const &rhs) const
            {
                return LilSparseMatrix<ScalarT>::operator==(rhs);
            }

            // necessary?
            bool operator!=(Matrix const &rhs) const
            {
                return LilSparseMatrix<ScalarT>::operator!=(rhs);
            }
        };
    }
}

// HACK
#include <graphblas/platforms/optimized_sequential/utility.hpp>
