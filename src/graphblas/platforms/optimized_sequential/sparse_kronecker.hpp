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

#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <iterator>
#include <iostream>
#include <chrono>

#include <graphblas/detail/logging.h>
#include <graphblas/types.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        /// Implementation of 4.3.11 kronecker: Matrix kronecker product
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void kronecker(CMatrixT            &C,
                              MMatrixT    const   &M,
                              AccumT      const   &accum,
                              BinaryOpT            op,
                              AMatrixT    const   &A,
                              BMatrixT    const   &B,
                              OutputControlEnum    outp)
        {
            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            IndexType ncol_A(A.ncols());
            IndexType nrow_B(B.nrows());
            IndexType ncol_B(B.ncols());
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;

            // =================================================================
            // Do the basic product work with the binaryop.
            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<BScalarType>()));
            LilSparseMatrix<TScalarType> T(nrow_C, ncol_C);
            typedef typename LilSparseMatrix<TScalarType>::RowType TRowType;

            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a row of the result at a time
                TRowType T_row;
                for (IndexType row_idxA = 0; row_idxA < nrow_A; ++row_idxA)
                {
                    auto const &A_row(A.getRow(row_idxA));
                    if (A_row.empty()) continue;

                    for (IndexType row_idxB = 0; row_idxB < nrow_B; ++row_idxB)
                    {
                        auto const &B_row(B.getRow(row_idxB));
                        if (B_row.empty()) continue;

                        T_row.clear();
                        IndexType T_row_idx(row_idxA*nrow_B + row_idxB);

                        for (auto &a_i : A_row)
                        {
                            IndexType col_idxA = std::get<0>(a_i);
                            AScalarType val_A = std::get<1>(a_i);

                            for (auto &b_i : B_row)
                            {
                                TScalarType T_val(op(val_A, std::get<1>(b_i)));
                                T_row.push_back(
                                    std::make_tuple(
                                        col_idxA*ncol_B + std::get<0>(b_i), T_val));
                            }
                        }

                        T.setRow(T_row_idx, T_row);
                    }
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<CScalarType>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

        }
    } // backend
} // GraphBLAS
