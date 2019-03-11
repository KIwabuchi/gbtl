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

/**
 * Implementation of sparse mxm for the sequential (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_MXM_ATBT_HPP
#define GB_SEQUENTIAL_SPARSE_MXM_ATBT_HPP

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
        // Compute C' = (A'*B')' = B*A, assuming C, B, and A are unique
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void ATBT_NoMask_NoAccum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            C.clear();
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB
                T_row.clear();
                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                //C.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    CScalarT  c_ji(static_cast<CScalarT>(std::get<1>(t)));

                    C[j].push_back(std::make_pair(i, c_ji));
                }
            }
            C.recomputeNvals();
        }
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_NoAccum_ATBT IN PROGRESS.\n";

            // C = (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================
            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;

            if (((void*)&C == (void*)&A) || ((void*)&C == (void*)&B))
            {
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(Ctmp, semiring, A, B);

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    C.setRow(i, Ctmp[i]);
                }
            }
            else
            {
                ATBT_NoMask_NoAccum_kernel(C, semiring, A, B);
            }
            C.recomputeNvals();

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_ATBT IN PROGRESS.\n";

            // C = C + (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================
            typedef typename SemiringT::result_type D3ScalarType;

            if (((void*)&C == (void*)&B) || ((void*)&C == (void*)&A))
            {
                // create temporary to prevent overwrite of inputs
                // T = A' +.* B'
                LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(T, semiring, A, B);
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T[i], accum);
                }
            }
            else
            {
                typename LilSparseMatrix<D3ScalarType>::RowType T_row;
                LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());
                ATBT_NoMask_NoAccum_kernel(T, semiring, A, B);

                // accumulate
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T[i], accum);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_NoAccum_ATBT IN PROGRESS.\n";

            // C<M,z> = A +.* B
            //        =               [M .* (A' +.* B')], z = "replace"
            //        = [!M .* C]  U  [M .* (A' +.* B')], z = "merge"
            // short circuit conditions
            if (replace_flag &&
                ((A.nvals() == 0) || (B.nvals() == 0) || (M.nvals() == 0)))
            {
                C.clear();
                return;
            }
            else if (!replace_flag && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            typename LilSparseMatrix<CScalarT>::RowType Z_row, C_row;
            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z = M[i] .* T[i]
                Z_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (advance_and_check_mask_iterator(m_it, M[i].end(), j))
                    {
                        Z_row.push_back(
                            std::make_tuple(
                                j, static_cast<CScalarT>(std::get<1>(t_ij))));
                    }
                }

                if (replace_flag)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = !M[i].*C[i] U Z_row
                    C_row.clear();
                    masked_merge(C_row, M[i], false, C[i], Z_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_ATBT IN PROGRESS.\n";

            // C<M,z> = C + (A' +.* B')
            //        =               [M .* [C + (A' +.* B')]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A' +.* B')]], z = "merge"
            // short circuit conditions
            if (replace_flag && (M.nvals() == 0))
            {
                C.clear();
                return;
            }
            else if (!replace_flag && (M.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AccumT::result_type ZScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            // compute transpose T' = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // T_row = M[i] .* T[i]
                T_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (advance_and_check_mask_iterator(m_it, M[i].end(), j))
                    {
                        T_row.push_back(t_ij);
                    }
                }

                // Z[i] = (M[i] .* C[i]) + T[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], false, accum, C[i], T_row);

                if (!replace_flag) /* z = merge */
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
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_NoAccum_ATBT IN PROGRESS.\n";

            // C<M,z> = A +.* B
            //        =               [M .* (A' +.* B')], z = "replace"
            //        = [!M .* C]  U  [M .* (A' +.* B')], z = "merge"
            // short circuit conditions
            if (replace_flag && ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            // compute transpose T = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            typename LilSparseMatrix<CScalarT>::RowType Z_row, C_row;
            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z = !M[i] .* T[i]
                Z_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (!advance_and_check_mask_iterator(m_it, M[i].end(), j))
                    {
                        Z_row.push_back(
                            std::make_tuple(
                                j, static_cast<CScalarT>(std::get<1>(t_ij))));
                    }
                }

                if (replace_flag)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = M[i].*C[i] U Z_row
                    C_row.clear();
                    masked_merge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);
                }
            }
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_Accum_ATBT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_ATBT IN PROGRESS.\n";

            // C<!M,z> = C + (A' +.* B')
            //         =              [!M .* [C + (A' +.* B')]], z = "replace"
            //         = [M .* C]  U  [!M .* [C + (A' +.* B')]], z = "merge"
            // short circuit conditions
            if (replace_flag &&
                (M.nvals() == 0) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                return; // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AccumT::result_type ZScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            // compute transpose T' = B +.* A (one row at a time and transpose)
            for (IndexType i = 0; i < B.nrows(); ++i)
            {
                // this part is same as sparse_mxm_NoMask_NoAccum_AB swapping
                // A and B and computing the transpose of C.
                T_row.clear();

                for (auto const &Bi_elt : B[i])
                {
                    IndexType    k(std::get<0>(Bi_elt));
                    AScalarT  b_ik(std::get<1>(Bi_elt));

                    if (A[k].empty()) continue;

                    // T[i] += (b_ik*A[k])  // must reduce in D3
                    axpy(T_row, semiring, b_ik, A[k]);
                }

                // Transpose the result
                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    T[j].push_back(std::make_pair(i, std::get<1>(t)));
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // T_row = M[i] .* T[i]
                T_row.clear();
                auto m_it = M[i].begin();
                for (auto t_ij : T[i])
                {
                    IndexType j(std::get<0>(t_ij));
                    if (!advance_and_check_mask_iterator(m_it, M[i].end(), j))
                    {
                        T_row.push_back(t_ij);
                    }
                }

                // Z[i] = (M[i] .* C[i]) + T[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], true, accum, C[i], T_row);

                if (!replace_flag) /* z = merge */
                {
                    // C[i]  = [!M .* C]  U  Z[i]
                    C_row.clear();
                    masked_merge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
                else // z = replace
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
            }
        }

    } // backend
} // GraphBLAS

#endif
