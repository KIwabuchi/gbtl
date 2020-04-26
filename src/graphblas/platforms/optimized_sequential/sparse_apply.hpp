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
#include <graphblas/types.hpp>
#include <graphblas/exceptions.hpp>
#include <graphblas/algebra.hpp>

#include "sparse_helpers.hpp"
#include "LilSparseMatrix.hpp"

//******************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //**********************************************************************
        // Implementation of 4.3.8.1 Vector variant of Apply
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename UVectorT,
                 typename ...WTagsT>
        inline void apply(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            UnaryOpT                                         op,
            UVectorT                                  const &u,
            OutputControlEnum                                outp)
        {
            // =================================================================
            // Apply the unary operator from A into T.
            typedef typename UVectorT::ScalarType        UScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(val));
                }
            }

            GRB_LOG_VERBOSE("t: " << t_contents);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            GRB_LOG_VERBOSE("z: " << z_contents);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z_contents, mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.8.2 Matrix variant of Apply
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply(
            GraphBLAS::backend::Matrix<CScalarT, CTagsT...> &C,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            UnaryOpT                                         op,
            AMatrixT                                  const &A,
            OutputControlEnum                                outp)
        {
            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            using TScalarType = decltype(op(std::declval<AScalarType>()));
            typedef std::vector<std::tuple<IndexType,TScalarType> > TRowType;

            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            LilSparseMatrix<TScalarType> T(nrows, ncols);
            TRowType t_row;

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                auto a_row = A.getRow(row_idx);
                if (!a_row.empty())
                {
                    t_row.clear();

                    for (auto&& [a_idx, a_val] : a_row) {
                        t_row.emplace_back(a_idx, op(a_val));
                    }

                    if (!t_row.empty())
                        T.setRow(row_idx, t_row);
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.8.3 Vector variant of Apply w/ binaryop+bind1st
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename ValueT,
                 typename UVectorT,
                 typename ...WTagsT>
        inline void apply_binop_1st(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            ValueT                                    const &val,
            UVectorT                                  const &u,
            OutputControlEnum                                outp)
        {
            // =================================================================
            // Apply the binary operator to u and val and store into T.
            // This is really the guts of what makes this special.
            typedef typename UVectorT::ScalarType         UScalarType;
            using TScalarType = decltype(op(std::declval<ValueT>(),
                                            std::declval<UScalarType>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, u_val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(val, u_val));
                }
            }

            GRB_LOG_VERBOSE("t: " << t_contents);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            GRB_LOG_VERBOSE("z: " << z_contents);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z_contents, mask, outp);
        }

        //**********************************************************************
        // Implementation of 4.3.8.3 Vector variant of Apply w/ binaryop+bind2nd
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename UVectorT,
                 typename ValueT,
                 typename ...WTagsT>
        inline void apply_binop_2nd(
            GraphBLAS::backend::Vector<WScalarT, WTagsT...> &w,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            UVectorT                                  const &u,
            ValueT                                    const &val,
            OutputControlEnum                                outp)
        {
            // =================================================================
            // Apply the binary operator to u and val and store into T.
            // This is really the guts of what makes this special.
            typedef typename UVectorT::ScalarType         UScalarType;
            using TScalarType = decltype(op(std::declval<UScalarType>(),
                                            std::declval<ValueT>()));
            std::vector<std::tuple<IndexType,TScalarType> > t_contents;

            if (u.nvals() > 0)
            {
                for (auto&& [idx, u_val] : u.getContents()) {
                    t_contents.emplace_back(idx, op(u_val, val));
                }
            }

            GRB_LOG_VERBOSE("t: " << t_contents);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType,ZScalarType> > z_contents;
            ewise_or_opt_accum_1D(z_contents, w, t_contents, accum);

            GRB_LOG_VERBOSE("z: " << z_contents);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask_1D(w, z_contents, mask, outp);
        }


        //**********************************************************************
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind2nd
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename ValueT,
                 typename AMatrixT,
                 typename ...CTagsT>
        inline void apply_binop_1st(
            GraphBLAS::backend::Matrix<CScalarT, CTagsT...> &C,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            ValueT                                    const &val,
            AMatrixT                                  const &A,
            OutputControlEnum                                outp)
        {
            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            using TScalarType = decltype(op(std::declval<ValueT>(),
                                            std::declval<AScalarType>()));
            typedef std::vector<std::tuple<IndexType,TScalarType> > TRowType;


            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            LilSparseMatrix<TScalarType> T(nrows, ncols);
            TRowType t_row;

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                auto a_row = A.getRow(row_idx);
                if (!a_row.empty())
                {
                    t_row.clear();

                    for (auto&& [a_idx, a_val] : a_row) {
                        t_row.emplace_back(a_idx, op(val, a_val));
                    }

                    if (!t_row.empty())
                        T.setRow(row_idx, t_row);
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
        }


        //**********************************************************************
        // Implementation of 4.3.8.4 Matrix variant of Apply w/ binaryop+bind2nd
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename AMatrixT,
                 typename ValueT,
                 typename ...CTagsT>
        inline void apply_binop_2nd(
            GraphBLAS::backend::Matrix<CScalarT, CTagsT...> &C,
            MaskT                                     const &mask,
            AccumT                                    const &accum,
            BinaryOpT                                        op,
            AMatrixT                                  const &A,
            ValueT                                    const &val,
            OutputControlEnum                                outp)
        {
            typedef typename AMatrixT::ScalarType                   AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > ARowType;

            using TScalarType = decltype(op(std::declval<AScalarType>(),
                                            std::declval<ValueT>()));
            typedef std::vector<std::tuple<IndexType,TScalarType> > TRowType;


            IndexType nrows(A.nrows());
            IndexType ncols(A.ncols());

            // =================================================================
            // Apply the unary operator from A into T.
            // This is really the guts of what makes this special.
            LilSparseMatrix<TScalarType> T(nrows, ncols);
            TRowType t_row;

            for (IndexType row_idx = 0; row_idx < A.nrows(); ++row_idx)
            {
                auto a_row = A.getRow(row_idx);
                if (!a_row.empty())
                {
                    t_row.clear();
                    for (auto&& [a_idx, a_val] : a_row) {
                        t_row.emplace_back(a_idx, op(a_val, val));
                    }

                    if (!t_row.empty())
                        T.setRow(row_idx, t_row);
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate T via C into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            LilSparseMatrix<ZScalarType> Z(nrows, ncols);
            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, mask, outp);
        }
    }
}
