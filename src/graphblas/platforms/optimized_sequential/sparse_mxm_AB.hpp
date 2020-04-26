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
        //**********************************************************************

        //**********************************************************************
        // Perform C = A*B where C, A and B must all be unique
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_NoMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += (a_ik*B[k])  // must reduce in D3
                    axpy(T_row, semiring, a_ik, B[k]);
                }

                // C[i] = T[i]
                C.setRow(i, T_row);  // set even if it is empty.
            }
        }

        //**********************************************************************
        // Perform C = C + AB, where C, A, and B must all be unique
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_NoMask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += (a_ik*B[k])  // must reduce in D3
                    axpy(T_row, semiring, a_ik, B[k]);
                }

                if (!T_row.empty())
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T_row, accum);
                }
            }
        }

        //**********************************************************************
        // Perform C<M,z> = A +.* B where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_Mask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                T_row.clear();

                // don't compute row if mask row is empty
                if (!M[i].empty())
                {
                    for (auto const &Ai_elt : A[i])
                    {
                        IndexType    k(std::get<0>(Ai_elt));
                        AScalarT  a_ik(std::get<1>(Ai_elt));

                        if (B[k].empty()) continue;

                        // T[i] += M[i] .* a_ik*B[k]
                        masked_axpy(T_row, M[i], false, semiring, a_ik, B[k]);
                    }
                }

                if (!(outp == REPLACE))
                {
                    // C[i] = [!M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    masked_merge(C_row, M[i], false, C[i], T_row);
                    C.setRow(i, C_row);
                }
                else
                {
                    // C[i] = T[i], z = "replace"
                    C.setRow(i, T_row);  // set even if it is empty.
                }
            }
        }

        //**********************************************************************
        // Perform C<M,z> = C + (A +.* B) where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_Mask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typedef decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>())) ZScalarType;

            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                T_row.clear();
                if (!M[i].empty())
                {
                    for (auto const &Ai_elt : A[i])
                    {
                        IndexType    k(std::get<0>(Ai_elt));
                        AScalarT  a_ik(std::get<1>(Ai_elt));

                        if (B[k].empty()) continue;

                        // T[i] += M[i] .* a_ik*B[k]
                        masked_axpy(T_row, M[i], false, semiring, a_ik, B[k]);
                    }
                }

                // Z[i] = (M .* C) + T[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], false, accum, C[i], T_row);

                if (!(outp == REPLACE)) /* z = merge */
                {
                    // C[i]  = [!M .* C]  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row, M[i], false, C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
                else // z = replace
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
            }
        }

        //**********************************************************************
        // Perform C<!M,z> = (A +.* B) where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_CompMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {

            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType     Z_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                // if M[i] is empty it is like NoMask_NoAccum

                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += !M[i] .* (a_ik*B[k])  // must reduce in D3
                    masked_axpy(T_row, M[i], true, semiring, a_ik, B[k]);
                }

                if ((outp == REPLACE) || M[i].empty())
                {
                    // C[i] = T[i]
                    C.setRow(i, T_row);  // set even if it is empty.
                }
                else
                {
                    Z_row.clear();
                    // Z[i] = (M[i] .* C[i]) U T[i]
                    masked_merge(Z_row, M[i], true, C[i], T_row);
                    C.setRow(i, Z_row);
                }
            }
        }

        //**********************************************************************
        // Perform C<!M,z> = C + (A +.* B) where A, B, M, and C are all unique
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void AB_CompMask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType     Z_row; // domain okay?
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                // if M[i] is empty it is like NoMask_NoAccum

                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += !M[i] .* (a_ik*B[k])  // must reduce in D3
                    masked_axpy(T_row, M[i], true, semiring, a_ik, B[k]);
                }

                // Z[i] = (!M[i] .* C[i]) + T[i], where T[i] is masked by !M[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], true, accum, C[i], T_row);

                if ((outp == REPLACE) || M[i].empty())
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = [M[i] .* C[i]]  U  Z[i], where Z is disjoint from M
                    C_row.clear();  // TODO: is an extra vector necessary?
                    masked_merge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);
                }

            }
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_AB(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            // C = A +.* B
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);
                C.swap(Ctmp);
            }
            else
            {
                AB_NoMask_NoAccum_kernel(C, semiring, A, B);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_AB(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            // C = C + (A +.* B)
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    C.mergeRow(i, Ctmp[i], accum);
                }
            }
            else
            {
                AB_NoMask_Accum_kernel(C, accum, semiring, A, B);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_AB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<M,z> = A +.* B
            //        =               [M .* (A +.* B)], z = "replace"
            //        = [!M .* C]  U  [M .* (A +.* B)], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) &&
                ((A.nvals() == 0) || (B.nvals() == 0) || (M.nvals() == 0)))
            {
                C.clear();
                return;
            }
            else if (!(outp == REPLACE) && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_Mask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                if ((outp == REPLACE))
                {
                    C.swap(Ctmp);
                }
                else
                {
                    typename LilSparseMatrix<CScalarT>::RowType C_row;
                    for (IndexType i = 0; i < C.nrows(); ++i)
                    {
                        // C[i] = [!M .* C]  U  T[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row, M[i], false, C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_Mask_NoAccum_kernel(C, M, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_Accum_AB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<M,z> = C + (A +.* B)
            //        =               [M .* [C + (A +.* B)]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B)]], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) && (M.nvals() == 0))
            {
                C.clear();
                return;
            }
            else if (!(outp == REPLACE) && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                typedef typename SemiringT::result_type D3ScalarType;
                typedef decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>())) ZScalarType;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<D3ScalarType> Ctmp(C.nrows(), C.ncols());
                AB_Mask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row, M[i], false, accum, C[i], Ctmp[i]);

                    if ((outp == REPLACE))
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row, M[i], false, C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_Mask_Accum_kernel(C, M, accum, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_AB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<!M,z> = A +.* B
            //        =              [!M .* (A +.* B)], z = "replace"
            //        = [M .* C]  U  [!M .* (A +.* B)], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) && ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_CompMask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                if ((outp == REPLACE))
                {
                    C.swap(Ctmp);
                }
                else
                {
                    typename LilSparseMatrix<CScalarT>::RowType C_row;
                    for (IndexType i = 0; i < C.nrows(); ++i)
                    {
                        // C[i] = [!M .* C]  U  T[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row, M[i], true, C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_CompMask_NoAccum_kernel(C, M, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_Accum_AB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            OutputControlEnum                outp)
        {
            // C<!M,z> = C + (A +.* B)
            //         =              [!M .* [C + (A +.* B)]], z = "replace"
            //         = [M .* C]  U  [!M .* [C + (A +.* B)]], z = "merge"
            // short circuit conditions
            if ((outp == REPLACE) &&
                (M.nvals() == 0) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                return; // do nothing
            }

            // =================================================================

            if ((void*)&C == (void*)&B)
            {
                typedef typename SemiringT::result_type D3ScalarType;
                typedef decltype(accum(std::declval<CScalarT>(),
                               std::declval<TScalarType>())) ZScalarType;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<D3ScalarType> Ctmp(C.nrows(), C.ncols());
                AB_CompMask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row, M[i], true, accum, C[i], Ctmp[i]);

                    if ((outp == REPLACE))
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        masked_merge(C_row, M[i], true, C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_CompMask_Accum_kernel(C, M, accum, semiring, A, B, outp);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

    } // backend
} // GraphBLAS
