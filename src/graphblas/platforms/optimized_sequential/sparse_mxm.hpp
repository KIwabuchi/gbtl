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

#ifndef GB_SEQUENTIAL_SPARSE_MXM_HPP
#define GB_SEQUENTIAL_SPARSE_MXM_HPP

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

#if 0
        //**********************************************************************
        /// 'sequential' Implementation of 4.3.1 mxm: Matrix-matrix multiply
        template<typename CMatrixT,
                 typename MMatrixT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void original_mxm(CMatrixT            &C,
                                 MMatrixT    const   &M,
                                 AccumT      const   &accum,
                                 SemiringT            op,
                                 AMatrixT    const   &A,
                                 BMatrixT    const   &B,
                                 bool                 replace_flag)
        {
            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            IndexType ncol_B(B.ncols());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create one rows of the result at a time
                for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                {
                    typename AMatrixT::RowType A_row(A.getRow(row_idx));
                    if (A_row.empty()) continue;

                    for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                    {
                        typename BMatrixT::ColType B_col(B.getCol(col_idx));

                        if (B_col.empty()) continue;

                        D3ScalarType T_val;
                        if (dot(T_val, A_row, B_col, op))
                        {
                            T[row_idx].push_back(
                                std::make_tuple(col_idx, T_val));
                        }
                    }
                }
                T.recomputeNvals();
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                D3ScalarType,
                typename AccumT::result_type>::type ZScalarType;
            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, M, replace_flag);

        } // mxm
#endif

        // *******************************************************************
        // Return true if iterator points to location with target_index;
        // otherwise returns the at the insertion point for target_index
        // which could be it_end.
        template <typename TupleIteratorT>
        bool advance_and_check_tuple_iterator(
            TupleIteratorT       &it,
            TupleIteratorT const &it_end,
            IndexType             target_index)
        {
            GRB_LOG_FN_BEGIN("advance_and_check_tuple_iterator: tgt = " << target_index);

            while ((it != it_end) && (std::get<0>(*it) < target_index))
            {
                ++it;
            }
            GRB_LOG_FN_END("advance_and_check_tuple_iterator target_found = "
                           << ((it != it_end) && (std::get<0>(*it) == target_index)));
            return ((it != it_end) && (std::get<0>(*it) == target_index));
        }

        // *******************************************************************
        // Only returns true if target index is found AND it evaluates to true
        template <typename TupleIteratorT>
        bool advance_and_check_mask_iterator(TupleIteratorT       &it,
                                             TupleIteratorT const &it_end,
                                             IndexType             target_index)
        {
            GRB_LOG_FN_BEGIN("advance_and_check_mask_iterator: tgt = " << target_index);

            bool tmp = (advance_and_check_tuple_iterator(it, it_end, target_index) &&
                        (static_cast<bool>(std::get<1>(*it))));

            GRB_LOG_FN_END("advance_and_check_mask_iterator res = " << tmp);
            return tmp;
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// c += a_ik*b[:]
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void axpy(std::vector<std::tuple<IndexType, CScalarT>>       &c,
                  SemiringT                                           semiring,
                  AScalarT                                            a,
                  std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("axpy");
            auto c_it = c.begin();

            for (auto const &b_elt : b)
            {
                IndexType   j(std::get<0>(b_elt));
                BScalarT  b_j(std::get<1>(b_elt));
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(c_it, c.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*c_it) = semiring.add(std::get<1>(*c_it), t_j);
                    ++c_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    c_it = c.insert(c_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++c_it;
                }
            }
            GRB_LOG_FN_END("axpy");
        }

        // *******************************************************************
        /// perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// c<[m[:]]> += a_ik*b[:]
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        void masked_axpy(std::vector<std::tuple<IndexType, CScalarT>>       &c,
                         std::vector<std::tuple<IndexType, MScalarT>> const &m,
                         bool                                                scmp_flag,
                         SemiringT                                           semiring,
                         AScalarT                                            a,
                         std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("masked_axpy");

            if (m.empty() && scmp_flag)
            {
                axpy(c, semiring, a, b);
                return;
            }

            auto c_it = c.begin();
            auto m_it = m.begin();

            for (auto const &b_elt : b)
            {
                IndexType    j(std::get<0>(b_elt));
                GRB_LOG_VERBOSE("j = " << j);

                // scan through M[i] to see if mask allows write.
                if (advance_and_check_mask_iterator(m_it, m.end(), j) == scmp_flag)
                {
                    GRB_LOG_VERBOSE("Skipping j = " << j);
                    continue;
                }

                BScalarT  b_j(std::get<1>(b_elt));

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advance_and_check_tuple_iterator(c_it, c.end(), j))
                {
                    GRB_LOG_VERBOSE("Accumulating");
                    std::get<1>(*c_it) = semiring.add(std::get<1>(*c_it), t_j);
                    ++c_it;
                }
                else
                {
                    GRB_LOG_VERBOSE("Inserting");
                    c_it = c.insert(c_it,
                                    std::make_tuple(j, static_cast<CScalarT>(t_j)));
                    ++c_it;
                }
            }
            GRB_LOG_FN_END("masked_axpy");
        }

        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>> (t assumed to be masked already)
        ///
        /// z = m ^ c (accum) t
        template<typename CScalarT,
                 typename AccumT,
                 typename MScalarT,
                 typename AScalarT,
                 typename BScalarT>
        void masked_accum(std::vector<std::tuple<IndexType, CScalarT>>       &z,
                          std::vector<std::tuple<IndexType, MScalarT>> const &m,
                          bool                                                scmp_flag,
                          AccumT                                       const &accum,
                          std::vector<std::tuple<IndexType, AScalarT>> const &c,
                          std::vector<std::tuple<IndexType, BScalarT>> const &t)
        {
            GRB_LOG_FN_BEGIN("masked_accum.v2");
            auto t_it = t.begin();
            auto m_it = m.begin();
            auto c_it = c.begin();

            // for each element of c find out if it is not in mask
            while ((t_it != t.end()) && (c_it != c.end()))
            {
                IndexType t_idx(std::get<0>(*t_it));
                IndexType c_idx(std::get<0>(*c_it));
                if (t_idx < c_idx)
                {
                    // t already masked
                    z.push_back(std::make_tuple(
                                    t_idx,
                                    static_cast<CScalarT>(std::get<1>(*t_it))));
                    ++t_it;
                }
                else if (c_idx < t_idx)
                {
                    if (advance_and_check_mask_iterator(m_it, m.end(), c_idx) != scmp_flag)
                    {
                        z.push_back(std::make_tuple(
                                        c_idx,
                                        static_cast<CScalarT>(std::get<1>(*c_it))));
                    }
                    ++c_it;
                }
                else
                {
                    z.push_back(
                        std::make_tuple(
                            t_idx,
                            static_cast<CScalarT>(accum(std::get<1>(*c_it),
                                                        std::get<1>(*t_it)))));
                    ++t_it;
                    ++c_it;
                }
            }

            while (t_it != t.end())
            {
                z.push_back(std::make_tuple(
                                std::get<0>(*t_it),
                                static_cast<CScalarT>(std::get<1>(*t_it))));
                ++t_it;
            }

            while (c_it != c.end())
            {
                IndexType c_idx(std::get<0>(*c_it));
                if (advance_and_check_mask_iterator(m_it, m.end(), c_idx) != scmp_flag)
                {
                    z.push_back(std::make_tuple(c_idx, std::get<1>(*c_it)));
                }
                ++c_it;
            }
            GRB_LOG_FN_END("masked_accum.v2");
        }

        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// c = (m ^ c) U (!m ^ a)  (assumes c and a are disjoint sets)
        template<typename CScalarT,
                 typename MScalarT,
                 typename AScalarT>
        void maskedMerge(std::vector<std::tuple<IndexType, CScalarT>>       &c,
                         std::vector<std::tuple<IndexType, MScalarT>> const &m,
                         bool                                                scmp_flag,
                         std::vector<std::tuple<IndexType, AScalarT>> const &a)
        {
            GRB_LOG_FN_BEGIN("maskedMerge");
            auto c_it = c.begin();
            auto m_it = m.begin();

            // for each element of the input matrix find out if it is not in mask
            for (auto const &a_j : a)
            {
                IndexType j(std::get<0>(a_j));

                // if mask has a stored value at j that evaluates to true then skip
                if (!advance_and_check_mask_iterator(m_it, m.end(), j) == scmp_flag)
                    continue;

                // ..otherwise merge
                advance_and_check_tuple_iterator(c_it, c.end(), j);
                c_it = c.insert(
                    c_it,
                    std::make_tuple(j, static_cast<CScalarT>(std::get<1>(a_j))));
                ++c_it;
            }
            GRB_LOG_FN_END("maskedMerge");
        }

        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>>
        ///
        /// if scmp_flag == false:
        ///    c = (!m ^ ci) U z, where z = (m ^ t);  i.e., union assumes disjoint sets
        /// else:
        ///   c =   (m ^ ci) U z, where z = (!m ^ t)
        template<typename CScalarT,
                 typename MScalarT,
                 typename ZScalarT>
        void maskedMerge(std::vector<std::tuple<IndexType, CScalarT>>       &c,
                         std::vector<std::tuple<IndexType, MScalarT>> const &m,
                         bool                                                scmp_flag,
                         std::vector<std::tuple<IndexType, CScalarT>> const &ci,
                         std::vector<std::tuple<IndexType, ZScalarT>> const &z)
        {
            GRB_LOG_FN_BEGIN("maskedMerge.v2");
            auto m_it = m.begin();
            auto c_it = ci.begin();
            auto z_it = z.begin();

            c.clear();

            IndexType next_z;
            for (auto const &elt : z)
            {
                next_z = std::get<0>(elt);
                while (c_it != ci.end() && (std::get<0>(*c_it) < next_z))
                {
                    IndexType next_c(std::get<0>(*c_it));
                    if (advance_and_check_mask_iterator(m_it, m.end(), next_c) == scmp_flag)
                    {
                        c.push_back(std::make_tuple(next_c, std::get<1>(*c_it)));
                    }
                    ++c_it;
                }
                c.push_back(std::make_tuple(next_z, static_cast<CScalarT>(std::get<1>(elt))));
            }


            while (c_it != ci.end() && (!z.empty() && (std::get<0>(*c_it) <= next_z)))
            {
                ++c_it;
            }

            while (c_it != ci.end())
            {
                IndexType next_c(std::get<0>(*c_it));
                if (advance_and_check_mask_iterator(m_it, m.end(), next_c) == scmp_flag)
                {
                    c.push_back(std::make_tuple(next_c, std::get<1>(*c_it)));
                }
                ++c_it;
            }

            GRB_LOG_FN_END("maskedMerge.v2");
        }


        //**********************************************************************
        // Perform C = A*B
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
            std::cout << "sparse_mxm_NoMask_NoAccum_AB COMPLETE.\n";

            // C = A +.* B
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================

            if (&C == &B)
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
            std::cout << "sparse_mxm_NoMask_Accum_AB COMPLETED.\n";

            // C = C + (A +.* B)
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            if (&C == &B)
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
        // Perform C<M,z> = A +.* B where A, B, M, and C are unique
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
            bool                             replace_flag)
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

                if (!replace_flag)
                {
                    // C[i] = [!M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], T_row);
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
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_NoAccum_AB COMPLETED (empty mask?).\n";

            // C<M,z> = A +.* B
            //        =               [M .* (A +.* B)], z = "replace"
            //        = [!M .* C]  U  [M .* (A +.* B)], z = "merge"
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

            if (&C == &B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_Mask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                if (replace_flag)
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
                        maskedMerge(C_row, M[i], false, C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_Mask_NoAccum_kernel(C, M, semiring, A, B, replace_flag);
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
        inline void AB_Mask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AccumT::result_type ZScalarType;

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

                if (!replace_flag) /* z = merge */
                {
                    // C[i]  = [!M .* C]  U  Z[i]
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], Z_row);
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
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_AB COMPLETE.\n";

            // C<M,z> = C + (A +.* B)
            //        =               [M .* [C + (A +.* B)]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B)]], z = "merge"
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

            if (&C == &B)
            {
                typedef typename SemiringT::result_type D3ScalarType;
                typedef typename AccumT::result_type ZScalarType;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<D3ScalarType> Ctmp(C.nrows(), C.ncols());
                AB_Mask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row, M[i], false, accum, C[i], Ctmp[i]);

                    if (replace_flag)
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        maskedMerge(C_row, M[i], false, C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_Mask_Accum_kernel(C, M, accum, semiring, A, B, replace_flag);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
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
            bool                             replace_flag)
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

                if (replace_flag || M[i].empty())
                {
                    // C[i] = T[i]
                    C.setRow(i, T_row);  // set even if it is empty.
                }
                else
                {
                    Z_row.clear();
                    // Z[i] = (M[i] .* C[i]) U T[i]
                    maskedMerge(Z_row, M[i], true, C[i], T_row);
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
        inline void sparse_mxm_CompMask_NoAccum_AB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_NoAccum_AB COMPLETED (empty mask?).\n";

            // C<!M,z> = A +.* B
            //        =              [!M .* (A +.* B)], z = "replace"
            //        = [M .* C]  U  [!M .* (A +.* B)], z = "merge"
            // short circuit conditions
            if (replace_flag && ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }

            // =================================================================

            if (&C == &B)
            {
                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<CScalarT> Ctmp(C.nrows(), C.ncols());
                AB_CompMask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                if (replace_flag)
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
                        maskedMerge(C_row, M[i], true, C[i], Ctmp[i]);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_CompMask_NoAccum_kernel(C, M, semiring, A, B, replace_flag);
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
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_AB COMPLETED.\n";

            // C<!M,z> = C + (A +.* B)
            //         =              [!M .* [C + (A +.* B)]], z = "replace"
            //         = [M .* C]  U  [!M .* [C + (A +.* B)]], z = "merge"
            // short circuit conditions
            if (replace_flag &&
                (M.nvals() == 0) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                return; // do nothing
            }

            // =================================================================

            if (&C == &B)
            {
                typedef typename SemiringT::result_type D3ScalarType;
                typedef typename AccumT::result_type ZScalarType;

                // create temporary to prevent overwrite of inputs
                LilSparseMatrix<D3ScalarType> Ctmp(C.nrows(), C.ncols());
                AB_CompMask_NoAccum_kernel(Ctmp, M, semiring, A, B, true);

                typename LilSparseMatrix<ZScalarType>::RowType  Z_row;

                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    Z_row.clear();
                    // Z[i] = (M .* C) + Ctmp[i]
                    masked_accum(Z_row, M[i], true, accum, C[i], Ctmp[i]);

                    if (replace_flag)
                    {
                        C.setRow(i, Z_row);
                    }
                    else
                    {
                        typename LilSparseMatrix<CScalarT>::RowType C_row;
                        // C[i] = [!M .* C]  U  Ctmp[i], z = "merge"
                        C_row.clear();
                        maskedMerge(C_row, M[i], true, C[i], Z_row);
                        C.setRow(i, C_row);
                    }
                }
            }
            else
            {
                AB_CompMask_Accum_kernel(C, M, accum, semiring, A, B, replace_flag);
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
        inline void AB_CompMask_Accum_kernel(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
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

                if (replace_flag || M[i].empty())
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // C[i] = [M[i] .* C[i]]  U  Z[i], where Z is disjoint from M
                    C_row.clear();  // TODO: is an extra vector necessary?
                    maskedMerge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);
                }

            }
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_NoAccum_ABT COMPLETED.\n";

            // C = (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                C_row.clear();
                if (A[i].empty()) continue;

                // fill row i of T
                for (IndexType j = 0; j < B.nrows(); ++j)
                {
                    if (B[j].empty()) continue;

                    D3ScalarType t_ij;

                    // Perform the dot product
                    // C[i][j] = T_ij = (CScalarT) (A[i] . B[j])
                    if (dot(t_ij, A[i], B[j], semiring))
                    {
                        C_row.push_back(
                            std::make_tuple(j, static_cast<CScalarT>(t_ij)));
                    }
                }

                C.setRow(i, C_row);  // set even if it is empty.
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_ABT COMPLETED.\n";

            // C = C + (A +.* B')
            // short circuit conditions?
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return;  // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            TRowType T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                if (A[i].empty()) continue;

                T_row.clear();

                // Compute row i of T
                // T[i] = (CScalarT) (A[i] *.+ B')
                for (IndexType j = 0; j < B.nrows(); ++j)
                {
                    if (B[j].empty()) continue;

                    D3ScalarType t_ij;

                    // Perform the dot product
                    // T[i][j] = (CScalarT) (A[i] . B[j])
                    if (dot(t_ij, A[i], B[j], semiring))
                    {
                        T_row.push_back(std::make_tuple(j, t_ij));
                    }
                }

                if (!T_row.empty())
                {
                    // C[i] = C[i] + T[i]
                    C.mergeRow(i, T_row, accum);
                }
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_NoAccum_ABT COMPLETED.\n";

            // C<M,z> = (A +.* B')
            //        =             [M .* (A +.* B')], z = replace
            //        = [!M .* C] U [M .* (A +.* B')], z = merge
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
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                T_row.clear();

                // T[i] = M[i] .* (A[i] dot B[j])
                if (!A[i].empty() && !M[i].empty())
                {
                    auto M_iter(M[i].begin());
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty() ||
                            !advance_and_check_mask_iterator(M_iter, M[i].end(), j))
                            continue;

                        // Perform the dot product
                        D3ScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.push_back(std::make_tuple(j, t_ij));
                        }
                    }
                }

                if (replace_flag)
                {
                    // C[i] = T[i], z = "replace"
                    C.setRow(i, T_row);
                }
                else /* merge */
                {
                    // C[i] = [!M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], T_row);
                    C.setRow(i, C_row);
                }
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
        inline void sparse_mxm_Mask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_ABT COMPLETED.\n";

            // C<M,z> = C + (A +.* B')
            //        =               [M .* [C + (A +.* B')]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B')]], z = "merge"
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
            typename LilSparseMatrix<ZScalarType>::RowType Z_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                Z_row.clear();

                if (!A[i].empty() && !M[i].empty())
                {
                    auto m_it(M[i].begin());
                    auto c_it(C[i].begin());

                    // Compute: T[i] = M[i] .* {C[i] + (A +.* B')[i]}
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        // See if B[j] has data and M[i] allows write.
                        if (B[j].empty() ||
                            !advance_and_check_mask_iterator(m_it, M[i].end(), j))
                            continue;

                        D3ScalarType t_ij;
                        bool have_c(advance_and_check_tuple_iterator(c_it, C[i].end(), j));

                        // Perform the dot product and accum if necessary
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            if (have_c)
                            {
                                Z_row.push_back(
                                    std::make_tuple(
                                        j, accum(std::get<1>(*c_it), t_ij)));
                            }
                            else
                            {
                                Z_row.push_back(
                                    std::make_tuple(
                                        j, static_cast<ZScalarType>(t_ij)));
                            }
                        }
                        else if (have_c)
                        {
                            Z_row.push_back(
                                std::make_tuple(
                                    j, static_cast<ZScalarType>(std::get<1>(*c_it))));
                        }
                    }
                }

                if (replace_flag)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // T[i] := (!M[i] .* C[i])  U  Z[i]
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_NoAccum_ABT COMPLETED.\n";

            // C<M,z> = (A +.* B')
            //        =            [!M .* (A +.* B')], z = replace
            //        = [M .* C] U [!M .* (A +.* B')], z = merge
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
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                T_row.clear();

                // T[i] = !M[i] .* (A[i] dot B[j])
                if (!A[i].empty()) // && !M[i].empty()) cannot do mask shortcut
                {
                    auto M_iter(M[i].begin());
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty() ||
                            advance_and_check_mask_iterator(M_iter, M[i].end(), j))
                            continue;

                        // Perform the dot product
                        D3ScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.push_back(std::make_tuple(j, t_ij));
                        }
                    }
                }

                if (replace_flag)
                {
                    // C[i] = T[i], z = "replace"
                    C.setRow(i, T_row);
                }
                else /* merge */
                {
                    // C[i] = [M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], true, C[i], T_row);
                    C.setRow(i, C_row);
                }
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
        inline void sparse_mxm_CompMask_Accum_ABT(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_ABT COMPLETED.\n";

            // C<M,z> = C + (A +.* B')
            //        =               [M .* [C + (A +.* B')]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B')]], z = "merge"
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
            typename LilSparseMatrix<ZScalarType>::RowType Z_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                Z_row.clear();

                if (!A[i].empty()) // && !M[i].empty()) cannot do mask shortcut
                {
                    auto m_it(M[i].begin());
                    auto c_it(C[i].begin());

                    // Compute: T[i] = M[i] .* {C[i] + (A +.* B')[i]}
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        // See if B[j] has data and M[i] allows write.
                        if (B[j].empty() ||
                            advance_and_check_mask_iterator(m_it, M[i].end(), j))
                            continue;

                        D3ScalarType t_ij;
                        bool have_c(advance_and_check_tuple_iterator(c_it, C[i].end(), j));

                        // Perform the dot product and accum if necessary
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            if (have_c)
                            {
                                Z_row.push_back(
                                    std::make_tuple(
                                        j, accum(std::get<1>(*c_it), t_ij)));
                            }
                            else
                            {
                                Z_row.push_back(
                                    std::make_tuple(
                                        j, static_cast<ZScalarType>(t_ij)));
                            }
                        }
                        else if (have_c)
                        {
                            Z_row.push_back(
                                std::make_tuple(
                                    j, static_cast<ZScalarType>(std::get<1>(*c_it))));
                        }
                    }
                }

                if (replace_flag)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    // T[i] := (M[i] .* C[i])  U  Z[i]
                    C_row.clear();
                    maskedMerge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);  // set even if it is empty.
                }
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        template<typename CScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_NoAccum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_NoAccum_ATB COMPLETE.\n";
            C.clear();

            // C = A +.* B
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return;
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    // T[i] += (a_ki*B[k])  // must reduce in D3, hence T.
                    axpy(T[i], semiring, a_ki, B[k]);
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                C.setRow(i, T[i]);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename AccumT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_NoMask_Accum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_ATB COMPLETED.\n";

            // C = C + (A +.* B)
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    // T[i] += (a_ki*B[k])  // must reduce in D3, hence T.
                    axpy(T[i], semiring, a_ki, B[k]);
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                if (!T[i].empty())
                    C.mergeRow(i, T[i], accum);
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_Mask_NoAccum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_NoAccum_ATB COMPLETED (empty mask?).\n";

            // C<M,z> = A' +.* B
            //        =               [M .* (A' +.* B)], z = "replace"
            //        = [!M .* C]  U  [M .* (A' +.* B)], z = "merge"
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
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    if (M[i].empty()) continue;

                    // T[i] += M[i] .* (a_ki*B[k])  // must reduce in D3, hence T.
                    masked_axpy(T[i], M[i], false, semiring, a_ki, B[k]);
                }
            }

            if (!replace_flag)
            {
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    // C[i] = [!M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], T[i]);
                    C.setRow(i, C_row);
                }
            }
            else
            {
                for (IndexType i = 0; i < C.nrows(); ++i)
                {
                    C.setRow(i, T[i]);
                }
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
        inline void sparse_mxm_Mask_Accum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_ATB COMPLETED\n";

            // C<M,z> = C + (A +.* B)
            //        =               [M .* [C + (A +.* B)]], z = "replace"
            //        = [!M .* C]  U  [M .* [C + (A +.* B)]], z = "merge"
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
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            typedef typename AccumT::result_type ZScalarType;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    if (M[i].empty()) continue;

                    // T[i] += M[i] .* (a_ki*B[k])  // must reduce in D3, hence T.
                    masked_axpy(T[i], M[i], false, semiring, a_ki, B[k]);
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z[i] = (M .* C) + T[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], false, accum, C[i], T[i]);

                if (!replace_flag) /* z = merge */
                {
                    // C[i] = [!M .* C]  U  Z[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], false, C[i], Z_row);
                    C.setRow(i, C_row);
                }
                else // z = replace
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
            }

            GRB_LOG_VERBOSE("C: " << C);
        }

        //**********************************************************************
        template<typename CScalarT,
                 typename MScalarT,
                 typename SemiringT,
                 typename AScalarT,
                 typename BScalarT>
        inline void sparse_mxm_CompMask_NoAccum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_NoAccum_ATB COMPLETED.\n";

            // C<M,z> = A' +.* B
            //        =              [!M .* (A' +.* B)], z = "replace"
            //        = [M .* C]  U  [!M .* (A' +.* B)], z = "merge"
            // short circuit conditions
            if (replace_flag && ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                C.clear();
                return;
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    // T[i] += !M[i] .* (a_ki*B[k])  // must reduce in D3, hence T.
                    masked_axpy(T[i], M[i], true, semiring, a_ki, B[k]);
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                if (replace_flag || M[i].empty())
                {
                    // C[i] = T[i]
                    C.setRow(i, T[i]);
                }
                else
                {
                    // C[i] = [M .* C]  U  T[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], true, C[i], T[i]);
                    C.setRow(i, C_row);
                }
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
        inline void sparse_mxm_CompMask_Accum_ATB(
            LilSparseMatrix<CScalarT>       &C,
            LilSparseMatrix<MScalarT> const &M,
            AccumT                    const &accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_ATB COMPLETED.\n";

            // C<M,z> = C + (A +.* B)
            //        =              [!M .* [C + (A +.* B)]], z = "replace"
            //        = [M .* C]  U  [!M .* [C + (A +.* B)]], z = "merge"
            // short circuit conditions
            if (replace_flag && (M.nvals() == 0) &&
                ((A.nvals() == 0) || (B.nvals() == 0)))
            {
                return; // do nothing
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

            typedef typename AccumT::result_type ZScalarType;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                for (auto const &Ak_elt : A[k])
                {
                    IndexType    i(std::get<0>(Ak_elt));
                    AScalarT  a_ki(std::get<1>(Ak_elt));

                    // T[i] += !M[i] .* (a_ki*B[k])  // must reduce in D3, hence T.
                    masked_axpy(T[i], M[i], true, semiring, a_ki, B[k]);
                }
            }

            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // Z[i] = (!M .* C) + T[i]
                Z_row.clear();
                masked_accum(Z_row, M[i], true, accum, C[i], T[i]);

                if (replace_flag || M[i].empty())
                {
                    // C[i] = Z[i]
                    C.setRow(i, Z_row);
                }
                else /* z = merge */
                {
                    // C[i] = [M .* C]  U  Z[i], z = "merge"
                    C_row.clear();
                    maskedMerge(C_row, M[i], true, C[i], Z_row);
                    C.setRow(i, C_row);
                }
            }

            GRB_LOG_VERBOSE("C: " << C);
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
            C.clear();

            // C = (A +.* B')
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return;
            }

            // =================================================================

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

/*
            for (IndexType j = 0; j < B.nrows(); ++j)
            {
                if (B[j].empty()) continue;

                for (auto const &Bj_elt : B[j])
                {
                    IndexType   k(std::get<0>(Bj_elt));
                    auto     b_jk(std::get<1>(Bj_elt));

                    if (A[k].empty()) continue;

                    for (auto const &Ak_elt : A[k])
                    {
                        IndexType   i(std::get<0>(Ak_elt));
                        auto     a_ki(std::get<1>(Ak_elt));

                        D3ScalarType tmp_val(semiring.mult(b_jk, a_ki));

                        if (!C[i].empty() && (std::get<0>(C[i].back()) == j))
                        {
                            std::get<1>(C[i].back()) =
                                semiring.add(std::get<1>(C[i].back()), tmp_val);
                        }
                        else
                        {
                            C[i].push_back(std::make_pair(j, tmp_val));
                        }
                    }
                }
            }
            C.recomputeNvals();
*/
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
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            LilSparseMatrix<D3ScalarType> T(C.nrows(), C.ncols());

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

                //T.setCol(i, T_row); // this is a push_back form of setCol
                for (auto const &t : T_row)
                {
                    IndexType j(std::get<0>(t));
                    CScalarT  t_ji(std::get<1>(t));

                    T[j].push_back(std::make_pair(i, t_ji));
                }
            }

            // accumulate
            for (IndexType i = 0; i < C.nrows(); ++i)
            {
                // C[i] = C[i] + T[i]
                C.mergeRow(i, T[i], accum);
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
            std::cout << "sparse_mxm_Mask_NoAccum_ATBT not implemented yet.\n";
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
            std::cout << "sparse_mxm_Mask_Accum_ATBT not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_NoAccum_ATBT not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_Accum_ATBT not implemented yet.\n";
        }

        //**********************************************************************
        //**********************************************************************
        //**********************************************************************

        //**********************************************************************
        /// Dispatch for 4.3.1 mxm: A * B
        //**********************************************************************
        template<class CMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        NoAccumulate const   &,
                        SR                    op,
                        AMat         const   &A,
                        BMat         const   &B,
                        bool                  replace_flag)
        {
            std::cout << "C := (A*B)" << std::endl;
            sparse_mxm_NoMask_NoAccum_AB(C, op, A, B);
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        Accum        const   &accum,
                        SR                    op,
                        AMat         const   &A,
                        BMat         const   &B,
                        bool                  replace_flag)
        {
            std::cout << "C := C + (A*B)" << std::endl;
            sparse_mxm_NoMask_Accum_AB(C, accum, op, A, B);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        MMat         const   &M,
                        NoAccumulate const   &,
                        SR                    op,
                        AMat         const   &A,
                        BMat         const   &B,
                        bool                  replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_AB(C, M, op, A, B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat            &C,
                        MMat    const   &M,
                        Accum   const   &accum,
                        SR               op,
                        AMat    const   &A,
                        BMat    const   &B,
                        bool             replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_Mask_Accum_AB(C, M, accum, op, A, B, replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_AB(C, strip_matrix_complement(M),
                                           op, A, B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        Accum                      const &accum,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_CompMask_Accum_AB(C, strip_matrix_complement(M), accum,
                                         op, A, B, replace_flag);
        }

        //**********************************************************************
        // Dispatch for 4.3.1 mxm: A * B'
        //**********************************************************************
        template<class CMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        NoMask              const &,
                        NoAccumulate        const &,
                        SR                         op,
                        AMat                const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C := (A*B')" << std::endl;
            sparse_mxm_NoMask_NoAccum_ABT(C, op, A, strip_transpose(B));
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        NoMask              const &,
                        Accum               const &accum,
                        SR                         op,
                        AMat                const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C := C + (A*B')" << std::endl;
            sparse_mxm_NoMask_Accum_ABT(C, accum, op, A, strip_transpose(B));
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        NoAccumulate        const &,
                        SR                         op,
                        AMat                const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ABT(C, M, op,
                                        A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        AMat                const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_Mask_Accum_ABT(C, M, accum, op,
                                      A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        TransposeView<BMat>        const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ABT(
                C, strip_matrix_complement(M), op,
                A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                &C,
                        MatrixComplementView<MMat>   const  &M,
                        Accum                        const  &accum,
                        SR                                   op,
                        AMat                         const  &A,
                        TransposeView<BMat>          const  &B,
                        bool                                 replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ABT(
                C, strip_matrix_complement(M), accum, op,
                A, strip_transpose(B), replace_flag);
        }

        //**********************************************************************
        // Dispatch for 4.3.1 mxm: A' * B
        //**********************************************************************
        template<class CMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                        &C,
                        NoMask              const   &,
                        NoAccumulate        const   &,
                        SR                           op,
                        TransposeView<AMat> const   &A,
                        BMat                const   &B,
                        bool                         replace_flag)
        {
            std::cout << "C := (A'*B)" << std::endl;
            sparse_mxm_NoMask_NoAccum_ATB(C, op, strip_transpose(A), B);
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                        &C,
                        NoMask              const   &,
                        Accum               const   &accum,
                        SR                           op,
                        TransposeView<AMat> const   &A,
                        BMat                const   &B,
                        bool                         replace_flag)
        {
            std::cout << "C := C + (A'*B)" << std::endl;
            sparse_mxm_NoMask_Accum_ATB(C, accum, op, strip_transpose(A), B);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                        &C,
                        MMat                const   &M,
                        NoAccumulate        const   &,
                        SR                           op,
                        TransposeView<AMat> const   &A,
                        BMat                const   &B,
                        bool                         replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_ATB(C, M, op,
                                        strip_transpose(A), B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        BMat                const &B,
                        bool                       replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_Mask_Accum_ATB(C, M, accum, op,
                                      strip_transpose(A), B, replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATB(
                C, strip_matrix_complement(M), op,
                strip_transpose(A), B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        Accum                      const &accum,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_CompMask_Accum_ATB(
                C, strip_matrix_complement(M), accum, op,
                strip_transpose(A), B, replace_flag);
        }

        //**********************************************************************
        // Dispatch for of 4.3.1 mxm: A' * B'
        //**********************************************************************
        template<class CMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        NoMask       const        &,
                        NoAccumulate const        &,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C := (A'*B')" << std::endl;
            sparse_mxm_NoMask_NoAccum_ATBT(C, op, strip_transpose(A),
                                           strip_transpose(B));
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        NoMask              const &,
                        Accum               const &accum,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C := C + (A'*B')" << std::endl;
            sparse_mxm_NoMask_Accum_ATBT(
                C, accum, op,
                strip_transpose(A), strip_transpose(B));
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        NoAccumulate        const &,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ATBT(
                C, M, op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        TransposeView<BMat> const &B,
                        bool                       replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_Mask_Accum_ATBT(
                C, M, accum, op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATBT(
                C, strip_matrix_complement(M), op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M,
                        Accum                      const &accum,
                        SR                                 op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        bool                              replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ATBT(
                C, strip_matrix_complement(M), accum, op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

    } // backend
} // GraphBLAS

#endif
