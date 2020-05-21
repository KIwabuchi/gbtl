/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
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
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"

//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //********************************************************************
        /// Implementation of 4.3.9.1 reduce: Standard Matrix to Vector variant:
        /// w<m,z> := op_j(A[:,j])
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        inline void reduce(WVectorT          &w,
                           MaskT       const &mask,
                           AccumT      const &accum,
                           BinaryOpT          op,
                           AMatrixT    const &A,
                           OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op_j(A[:,j]) (row reduce)");

            // =================================================================
            // Do the basic reduction work with the binary op
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType t_val;
                    if (reduction(t_val, A[row_idx], op))
                    {
                        t.emplace_back(row_idx, t_val);
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //********************************************************************
        /// Implementation of 4.3.9.1 reduce: Standard Matrix to Vector variant:
        /// w<m,z> := op_j(A'[:,j]) = op_j(A[j,:]) column reduce
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        inline void reduce(WVectorT                      &w,
                           MaskT                   const &mask,
                           AccumT                  const &accum,
                           BinaryOpT                      op,
                           TransposeView<AMatrixT> const &AT,
                           OutputControlEnum              outp)
        {
            GRB_LOG_VERBOSE("w<m,z> := op_j(A[:,j]) = op_j(A[j,:]) (col reduce)");
            auto const &A(strip_transpose(AT));

            // =================================================================
            // Do the basic reduction work with the binary op
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    if (!A[row_idx].empty())
                        xpey(t, A[row_idx], op);
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            using ZScalarType = typename std::conditional_t<
                std::is_same_v<AccumT, NoAccumulate>,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>;
            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //********************************************************************
        /// Implementation of 4.3.9.2 reduce: Vector to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UVectorT>
        inline void reduce_vector_to_scalar(ValueT         &val,
                                            AccumT   const &accum,
                                            MonoidT         op,
                                            UVectorT const &u)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            typedef typename UVectorT::ScalarType UScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>(),
                                            std::declval<UScalarType>()));
            typedef std::vector<std::tuple<IndexType,UScalarType> >  UColType;

            TScalarType t = op.identity();

            if (u.nvals() > 0)
            {
                UColType const u_col(u.getContents());

                reduction(t, u_col, op);
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

        //********************************************************************
        /// Implementation of 4.3.9.3 reduce: Matrix to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AMatrixT>
        inline void reduce_matrix_to_scalar(ValueT         &val,
                                            AccumT   const &accum,
                                            MonoidT         op,
                                            AMatrixT const &A)
        {
            // =================================================================
            // Do the basic reduction work with the monoid
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            TScalarType t = op.identity();

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType tmp;

                    if (!A[row_idx].empty())
                    {
                        if (reduction(tmp, A[row_idx], op)) // reduce each row
                        {
                            t = op(t, tmp); // reduce across rows
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

        //********************************************************************
        /// Implementation of 4.3.9.3 reduce: transpose Matrix to scalar variant
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename AMatrixT>
        inline void reduce_matrix_to_scalar(
            ValueT                        &val,
            AccumT                  const &accum,
            MonoidT                        op,
            TransposeView<AMatrixT> const &AT)
        {
            auto const &A(strip_transpose(AT));

            // =================================================================
            // Do the basic reduction work with the monoid
            using TScalarType =
                decltype(op(std::declval<typename AMatrixT::ScalarType>(),
                            std::declval<typename AMatrixT::ScalarType>()));
            TScalarType t = op.identity();

            if (A.nvals() > 0)
            {
                for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
                {
                    /// @todo There is something hinky with domains here.  How
                    /// does one perform the reduction in A domain but produce
                    /// partial results in D3(op)?
                    TScalarType tmp;

                    if (!A[row_idx].empty())
                    {
                        if (reduction(tmp, A[row_idx], op)) // reduce each row
                        {
                            t = op(t, tmp); // reduce across rows
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            // Type generator for z: D3(accum), or D(w) if no accum.
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<ValueT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            ZScalarType z;
            opt_accum_scalar(z, val, t, accum);

            // Copy Z into the final output
            val = z;
        }

    } // backend
} // GraphBLAS
