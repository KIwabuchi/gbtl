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
        /// Implementation of 4.3.3 mxv: Matrix-Vector variant
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void original_mxv(WVectorT          &w,
                                 MaskT       const &mask,
                                 AccumT      const &accum,
                                 SemiringT          op,
                                 AMatrixT    const &A,
                                 UVectorT    const &u,
                                 OutputControlEnum  outp)
        {
            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            typedef typename SemiringT::result_type TScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> >  ARowType;

            std::vector<std::tuple<IndexType, TScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                auto u_contents(u.getContents());
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    ARowType const &A_row(A.getRow(row_idx));

                    if (!A_row.empty())
                    {
                        TScalarType t_val;
                        if (dot(t_val, A_row, u_contents, op))
                        {
                            t.push_back(std::make_tuple(row_idx, t_val));
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<typename WVectorT::ScalarType>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UScalarT>
        inline void sparse_mxv_A(BitmapSparseVector<WScalarT>       &w,
                                 MaskT                        const &mask,
                                 AccumT                       const &accum,
                                 SemiringT                           op,
                                 LilSparseMatrix<AScalarT>    const &A,
                                 BitmapSparseVector<UScalarT> const &u,
                                 OutputControlEnum                   outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A +.*u");

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            typedef typename SemiringT::result_type TScalarType;
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < w.size(); ++row_idx)
                {
                    if (!A[row_idx].empty())
                    {
                        TScalarType t_val;
                        if (dot2(t_val, A[row_idx],
                                 u.get_bitmap(), u.get_vals(), u.nvals(), op))
                        {
                            t.push_back(std::make_tuple(row_idx, t_val));
                        }
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //**********************************************************************
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename UScalarT>
        inline void sparse_mxv_AT(BitmapSparseVector<WScalarT>       &w,
                                  MaskT                        const &mask,
                                  AccumT                       const &accum,
                                  SemiringT                           op,
                                  LilSparseMatrix<AScalarT>    const &A,
                                  BitmapSparseVector<UScalarT> const &u,
                                  OutputControlEnum                   outp)
        {
            GRB_LOG_VERBOSE("w<M,z> := A' +.* u");

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            typedef typename SemiringT::result_type TScalarType;
            std::vector<std::tuple<IndexType, TScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                for (IndexType row_idx = 0; row_idx < u.size(); ++row_idx)
                {
                    if (u.hasElement(row_idx) && !A[row_idx].empty())
                    {
                        axpy(t, op, u.extractElement(row_idx), A[row_idx]);
                    }
                }
            }

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                TScalarType,
                decltype(accum(std::declval<WScalarT>(),
                               std::declval<TScalarType>()))>::type
                ZScalarType;

            std::vector<std::tuple<IndexType, ZScalarType> > z;
            ewise_or_opt_accum_1D(z, w, t, accum);

            // =================================================================
            // Copy Z into the final output, w, considering mask and replace/merge
            write_with_opt_mask_1D(w, z, mask, outp);
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        /// Dispatch for 4.3.2 mxv: A * u
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT      const &accum,
                        SemiringT          op,
                        AMatrixT    const &A,
                        UVectorT    const &u,
                        OutputControlEnum  outp)
        {
            GRB_LOG_VERBOSE("C := (A*B)");
            //sparse_mxv_A(w, mask, accum, op, A, u, outp);
            original_mxv(w, mask, accum, op, A, u, outp);
        }

        //**********************************************************************
        /// Dispatch for 4.3.2 mxv: A' * u
        //**********************************************************************
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void mxv(WVectorT                       &w,
                        MaskT                    const &mask,
                        AccumT                   const &accum,
                        SemiringT                       op,
                        TransposeView<AMatrixT>  const &A,
                        UVectorT                 const &u,
                        OutputControlEnum               outp)
        {
            GRB_LOG_VERBOSE("C := (A*B)");
            sparse_mxv_AT(w, mask, accum, op, strip_transpose(A), u, outp);
        }

    } // backend
} // GraphBLAS
