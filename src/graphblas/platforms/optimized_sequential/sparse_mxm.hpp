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
        // *******************************************************************
        // Return true if iterator points to location with target_index;
        // otherwise returns the at the insertion point for target_index
        // which could be it_end.
        template <typename TupleIteratorT>
        bool advanceTupleIterator(
            TupleIteratorT &it, TupleIteratorT const &it_end, IndexType target_index)
        {
            GRB_LOG_FN_BEGIN("advanceTupleIterator: tgt = " << target_index);

            while ((it != it_end) && (std::get<0>(*it) < target_index))
            {
                ++it;
            }
            GRB_LOG_FN_END("advanceTupleIterator target_found = "
                           << ((it != it_end) && (std::get<0>(*it) == target_index)));
            return ((it != it_end) && (std::get<0>(*it) == target_index));
        }

        // *******************************************************************
        // Only returns true if target index is found AND it evaluates to true
        template <typename TupleIteratorT>
        bool advanceAndCheckMaskIterator(TupleIteratorT       &it,
                                         TupleIteratorT const &it_end,
                                         IndexType             target_index)
        {
            GRB_LOG_FN_BEGIN("advanceAndCheckMaskIterator: tgt = " << target_index);

            bool tmp = (advanceTupleIterator(it, it_end, target_index) &&
                        (static_cast<bool>(std::get<1>(*it))));

            GRB_LOG_FN_END("advanceAndCheckMaskIterator res = " << tmp);
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
                IndexType    j(std::get<0>(b_elt));
                BScalarT  b_j(std::get<1>(b_elt));
                GRB_LOG_VERBOSE("j = " << j);

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advanceTupleIterator(c_it, c.end(), j))
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
        void maskedAxpy(std::vector<std::tuple<IndexType, CScalarT>>       &c,
                        std::vector<std::tuple<IndexType, MScalarT>> const &m,
                        bool                                                scmp_flag,
                        SemiringT                                           semiring,
                        AScalarT                                            a,
                        std::vector<std::tuple<IndexType, BScalarT>> const &b)
        {
            GRB_LOG_FN_BEGIN("maskedAxpy");

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
                if (advanceAndCheckMaskIterator(m_it, m.end(), j) == scmp_flag)
                {
                    GRB_LOG_VERBOSE("Skipping j = " << j);
                    continue;
                }

                BScalarT  b_j(std::get<1>(b_elt));

                auto t_j(semiring.mult(a, b_j));
                GRB_LOG_VERBOSE("temp = " << t_j);

                // scan through C_row to find insert/merge point
                if (advanceTupleIterator(c_it, c.end(), j))
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
            GRB_LOG_FN_END("maskedAxpy");
        }

        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>> (note result is accumulated into rhs
        ///
        /// z = m ^ lhs (accum) rhs
        template<typename CScalarT,
                 typename AccumT,
                 typename MScalarT>
        void maskedAccum(std::vector<std::tuple<IndexType, CScalarT>>       &rhs,
                         std::vector<std::tuple<IndexType, MScalarT>> const &m,
                         bool                                                scmp_flag,
                         AccumT                                              op,
                         std::vector<std::tuple<IndexType, CScalarT>> const &lhs)
        {
            GRB_LOG_FN_BEGIN("maskedAccum");
            auto r_it = rhs.begin();
            auto m_it = m.begin();

            // for each element of the input matrix find out if it is not in mask
            for (auto const &l_elt : lhs)
            {
                IndexType l_idx(std::get<0>(l_elt));

                // scan through M[i] to see if mask allows write.
                if (advanceAndCheckMaskIterator(m_it, m.end(), l_idx) == scmp_flag)
                {
                    GRB_LOG_VERBOSE("Skipping l_idx = " << l_idx);
                    continue;
                }

                // if mask has a stored value at j that evaluates to true, ewiseadd
                // (using accum) with rhs and store result in rhs
                if (advanceTupleIterator(r_it, rhs.end(), l_idx))
                {
                    std::get<1>(*r_it) =
                        op(std::get<1>(l_elt) , std::get<1>(*r_it));
                }
                else
                {
                    // insert the value
                    r_it = rhs.insert(r_it, l_elt);
                    ++r_it;
                }

            }
            GRB_LOG_FN_END("maskedAccum");
        }

        /// Perform the following operation on sparse vectors implemented as
        /// vector<tuple<Index, value>> (t assumed to be masked already)
        ///
        /// z = m ^ c (accum) t
        template<typename CScalarT,
                 typename AccumT,
                 typename MScalarT,
                 typename TScalarT>
        void maskedAccum(std::vector<std::tuple<IndexType, CScalarT>>       &z,
                         std::vector<std::tuple<IndexType, MScalarT>> const &m,
                         bool                                                scmp_flag,
                         AccumT                                              op,
                         std::vector<std::tuple<IndexType, CScalarT>> const &c,
                         std::vector<std::tuple<IndexType, TScalarT>> const &t)
        {
            GRB_LOG_FN_BEGIN("maskedAccum.v2");
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
                    if (advanceAndCheckMaskIterator(m_it, m.end(), c_idx) != scmp_flag)
                    {
                        z.push_back(std::make_tuple(
                                        c_idx, std::get<1>(*c_it)));
                    }
                    ++c_it;
                }
                else
                {
                    z.push_back(
                        std::make_tuple(
                            t_idx,
                            static_cast<CScalarT>(op(std::get<1>(*c_it),
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
                if (advanceAndCheckMaskIterator(m_it, m.end(), c_idx) != scmp_flag)
                {
                    z.push_back(std::make_tuple(c_idx, std::get<1>(*c_it)));
                }
                ++c_it;
            }
            GRB_LOG_FN_END("maskedAccum.v2");
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
                if (!advanceAndCheckMaskIterator(m_it, m.end(), j) == scmp_flag)
                    continue;

                // ..otherwise merge
                advanceTupleIterator(c_it, c.end(), j);
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
        /// c = (!m ^ ci) U z, where z = (m ^ t)  where union assumes disjoint sets
        ///  or
        /// c = (m ^ ci)  U z, where z = (!m ^ t) if scmp_flag==true
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
                    if (advanceAndCheckMaskIterator(m_it, m.end(), next_c) == scmp_flag)
                    {
                        c.push_back(std::make_tuple(next_c, std::get<1>(*c_it)));
                    }
                    ++c_it;
                }
                c.push_back(std::make_tuple(next_z, static_cast<CScalarT>(std::get<1>(elt))));
            }

            while (c_it != ci.end() && ((std::get<0>(*c_it) <= next_z)))
            {
                ++c_it;
            }

            while (c_it != ci.end())
            {
                IndexType next_c(std::get<0>(*c_it));
                if (advanceAndCheckMaskIterator(m_it, m.end(), next_c) == scmp_flag)
                {
                    c.push_back(std::make_tuple(next_c, std::get<1>(*c_it)));
                }
                ++c_it;
            }

            GRB_LOG_FN_END("maskedMerge.v2");
        }


#if 1
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
                                 AccumT               accum,
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

                C.setRow(i, T_row);  // set even if it is empty.
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_AB IN PROGRESS.\n";

            // C = C + (A +.* B)
            // short circuit conditions
            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return; // do nothing
            }

            // =================================================================

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
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_NoAccum_AB IN PROGRESS (empty mask?).\n";

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

            typedef typename SemiringT::result_type D3ScalarType;
            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<CScalarT>::RowType C_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                if (M[i].empty()) continue; // nothing to do if row of mask is empty

                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += M[i] .* a_ik*B[k]
                    maskedAxpy(T_row, M[i], false, semiring, a_ik, B[k]);
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
            AccumT                           accum,
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

            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AccumT::result_type ZScalarType;

            typename LilSparseMatrix<D3ScalarType>::RowType T_row;
            typename LilSparseMatrix<ZScalarType>::RowType  Z_row;
            typename LilSparseMatrix<CScalarT>::RowType     C_row;

            for (IndexType i = 0; i < A.nrows(); ++i) // compute row i of answer
            {
                if (M[i].empty()) continue; // nothing to do if row of mask is empty

                T_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    // T[i] += M[i] .* a_ik*B[k]
                    maskedAxpy(T_row, M[i], false, semiring, a_ik, B[k]);
                }

                // Z[i] = (M .* C) + T[i]
                Z_row.clear();
                maskedAccum(Z_row, M[i], false, accum, C[i], T_row);

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
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_NoAccum_AB IN PROGRESS (empty mask?).\n";

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
                    maskedAxpy(T_row, M[i], true, semiring, a_ik, B[k]);
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_AB IN PROGRESS.\n";

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
                    maskedAxpy(T_row, M[i], true, semiring, a_ik, B[k]);
                }

                // Z[i] = (!M[i] .* C[i]) + T[i], where T[i] is masked buy !M[i]
                Z_row.clear();
                maskedAccum(Z_row, M[i], true, accum, C[i], T_row);

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

            GRB_LOG_VERBOSE("C: " << C);
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
            std::cout << "sparse_mxm_NoMask_NoAccum_ABT IN PROGRESS.\n";

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

            // Build this completely based on the semiring
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
            std::cout << "sparse_mxm_NoMask_Accum_ABT IN PROGRESS.\n";

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

            // Build this completely based on the semiring
            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                if (A[i].empty()) continue;

                T_row.clear();

                // Compute row i of T
                for (IndexType j = 0; j < B.nrows(); ++j)
                {
                    if (B[j].empty()) continue;

                    D3ScalarType t_ij;

                    // Perform the dot product
                    if (dot(t_ij, A[i], B[j], semiring))
                    {
                        T_row.push_back(std::make_tuple(j, t_ij));
                    }
                }

                if (!T_row.empty())
                {
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
            std::cout << "sparse_mxm_Mask_NoAccum_ABT IN PROGRESS.\n";

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
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            TRowType T_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                T_row.clear();

                // fill row i of T
                if (!A[i].empty())
                {
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty()) continue;

                        // Perform the dot product
                        D3ScalarType t_ij;
                        if (dot(t_ij, A[i], B[j], semiring))
                        {
                            T_row.push_back(std::make_tuple(j, t_ij));
                        }
                    }
                }

                if (!replace_flag)
                {
                    maskedMerge(T_row, M[i], false, C[i]);
                }

                C.setRow(i, T_row);
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_ABT IN PROGRESS.\n";

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
            typedef std::vector<std::tuple<IndexType,ZScalarType> > ZRowType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            ZRowType Z_row;

            for (IndexType i = 0; i < A.nrows(); ++i)
            {
                Z_row.clear();

                if (!A[i].empty() && !M[i].empty())
                {
                    // Compute: T[i] = M[i] .* {C[i] + (A +.* B')[i]}

                    auto m_it(M[i].begin());
                    auto c_it(C[i].begin());
                    for (IndexType j = 0; j < B.nrows(); ++j)
                    {
                        if (B[j].empty()) continue;

                        // Scan through M[i] to see if mask allows write.
                        if (advanceAndCheckMaskIterator(m_it, M[i].end(), j))
                        {
                            D3ScalarType t_ij;
                            bool have_c(advanceTupleIterator(c_it, C[i].end(), j));

                            // Perform the dot product and accum if necessary
                            if (dot(t_ij, A[i], B[j], semiring))
                            {
                                if (have_c)
                                {
                                    Z_row.push_back(
                                        std::make_tuple(j, accum(std::get<1>(*c_it), t_ij)));
                                }
                                else
                                {
                                    Z_row.push_back(
                                        std::make_tuple(j, static_cast<ZScalarType>(t_ij)));
                                }
                            }
                            else if (have_c)
                            {
                                Z_row.push_back(std::make_tuple(
                                                    j, static_cast<ZScalarType>(
                                                        std::get<1>(*c_it))));
                            }
                        }
                    }
                }

                if (replace_flag)
                {
                    C.setRow(i, Z_row);
                }
                else /* merge */
                {
                    CRowType C_row;
                    // T[i] := [!M .* C]  U  [M[i] .* (C[i] + T[i])], z = "merge"
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
            std::cout << "sparse_mxm_CompMask_NoAccum_ABT not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_Accum_ABT not implemented yet.\n";
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

            // C = A +.* B
            // short circuit conditions

            C.clear();

            if ((A.nvals() == 0) || (B.nvals() == 0))
            {
                return;
            }

            // =================================================================

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            TRowType Ci_tmp;

            for (IndexType k = 0; k < A.nrows(); ++k)
            {
                if (A[k].empty() || B[k].empty()) continue;

                auto Ak_it(A[k].begin());

                while (Ak_it != A[k].end())
                {
                    IndexType    i(std::get<0>(*Ak_it));
                    AScalarT  a_ki(std::get<1>(*Ak_it));

                    auto Bk_it(B[k].begin());

                    Ci_tmp.clear();

                    while (Bk_it != B[k].end())
                    {
                        IndexType j(std::get<0>(*Bk_it));
                        D3ScalarType t_kj(semiring.mult(a_ki, std::get<1>(*Bk_it)));
                        Ci_tmp.push_back(std::make_tuple(j, t_kj));
                        ++Bk_it;
                    }

                    C.mergeRow(i, Ci_tmp,
                               AdditiveMonoidFromSemiring<SemiringT>(semiring));
                    ++Ak_it;
                }
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_ATB not implemented yet.\n";
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
            std::cout << "sparse_mxm_Mask_NoAccum_ATB not implemented yet.\n";
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_Mask_Accum_ATB not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_NoAccum_ATB not implemented yet.\n";
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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B,
            bool                             replace_flag)
        {
            std::cout << "sparse_mxm_CompMask_Accum_ATB not implemented yet.\n";
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
            C.clear();
            if ((A.nvals() == 0) && (B.nvals() == 0)) return;

            // =================================================================
            // Build this completely based on the semiring
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

                        CScalarT tmp_val(semiring.mult(b_jk, a_ki));

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
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_ATBT not implemented yet.\n";
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
            AccumT                           accum,
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
            AccumT                           accum,
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
        inline void mxm(CMat                               &C,
                        MatrixComplementView<MMat> const   &M,
                        NoAccumulate               const   &,
                        SR                                  op,
                        AMat                      const   &A,
                        BMat                      const   &B,
                        bool                                replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_AB(C, strip_matrix_complement(M),
                                           op, A, B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                  &C,
                        MatrixComplementView<MMat>    const   &M,
                        Accum                         const   &accum,
                        SR                                     op,
                        AMat                         const   &A,
                        BMat                         const   &B,
                        bool                                   replace_flag)
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
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        NoAccumulate const   &,
                        SR                    op,
                        AMat                const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C := (A*B')" << std::endl;
            sparse_mxm_NoMask_NoAccum_ABT(C, op, A, strip_transpose(B));
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        Accum        const   &accum,
                        SR                    op,
                        AMat                const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C := C + (A*B')" << std::endl;
            sparse_mxm_NoMask_Accum_ABT(C, accum, op, A, strip_transpose(B));
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        MMat         const   &M,
                        NoAccumulate const   &,
                        SR                    op,
                        AMat                const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ABT(C, M, op,
                                        A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat            &C,
                        MMat    const   &M,
                        Accum   const   &accum,
                        SR               op,
                        AMat                const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_Mask_Accum_ABT(C, M, accum, op,
                                      A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                               &C,
                        MatrixComplementView<MMat> const   &M,
                        NoAccumulate               const   &,
                        SR                                  op,
                        AMat                      const   &A,
                        TransposeView<BMat>       const   &B,
                        bool                                replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ABT(
                C, strip_matrix_complement(M), op,
                A, strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                  &C,
                        MatrixComplementView<MMat>    const   &M,
                        Accum                         const   &accum,
                        SR                                     op,
                        AMat                         const   &A,
                        TransposeView<BMat>          const   &B,
                        bool                                   replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ABT(
                C, strip_matrix_complement(M), op, accum,
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
        inline void mxm(CMat                        &C,
                        MMat                const   &M,
                        Accum               const   &accum,
                        SR                           op,
                        TransposeView<AMat> const   &A,
                        BMat                const   &B,
                        bool                         replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_Mask_Accum_ATB(C, M, accum, op,
                                      strip_transpose(A), B, replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                               &C,
                        MatrixComplementView<MMat> const   &M,
                        NoAccumulate               const   &,
                        SR                                  op,
                        TransposeView<AMat>        const   &A,
                        BMat                       const   &B,
                        bool                                replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATB(
                C, strip_matrix_complement(M), op,
                strip_transpose(A), B, replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                               &C,
                        MatrixComplementView<MMat> const   &M,
                        Accum                      const   &accum,
                        SR                                  op,
                        TransposeView<AMat>        const   &A,
                        BMat                       const   &B,
                        bool                                replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_CompMask_Accum_ATB(
                C, strip_matrix_complement(M), op, accum,
                strip_transpose(A), B, replace_flag);
        }

        //**********************************************************************
        // Dispatch for of 4.3.1 mxm: A' * B'
        //**********************************************************************
        template<class CMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        NoAccumulate const   &,
                        SR                    op,
                        TransposeView<AMat> const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C := (A'*B')" << std::endl;
            sparse_mxm_NoMask_NoAccum_ATBT(C, op, strip_transpose(A),
                                           strip_transpose(B));
        }

        template<class CMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        NoMask       const   &,
                        Accum        const   &accum,
                        SR                    op,
                        TransposeView<AMat> const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C := C + (A'*B')" << std::endl;
            sparse_mxm_NoMask_Accum_ATBT(
                C, accum, op,
                strip_transpose(A), strip_transpose(B));
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                 &C,
                        MMat         const   &M,
                        NoAccumulate const   &,
                        SR                    op,
                        TransposeView<AMat> const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ATBT(
                C, M, op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat            &C,
                        MMat    const   &M,
                        Accum   const   &accum,
                        SR               op,
                        TransposeView<AMat> const   &A,
                        TransposeView<BMat> const   &B,
                        bool                          replace_flag)
        {
            std::cout << "C<M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_Mask_Accum_ATBT(
                C, M, accum, op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                               &C,
                        MatrixComplementView<MMat> const   &M,
                        NoAccumulate               const   &,
                        SR                                  op,
                        TransposeView<AMat>       const   &A,
                        TransposeView<BMat>       const   &B,
                        bool                                replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATBT(
                C, strip_matrix_complement(M), op,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                  &C,
                        MatrixComplementView<MMat>    const   &M,
                        Accum                         const   &accum,
                        SR                                     op,
                        TransposeView<AMat>          const   &A,
                        TransposeView<BMat>          const   &B,
                        bool                                   replace_flag)
        {
            std::cout << "C<!M" << (replace_flag ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ATBT(
                C, strip_matrix_complement(M), op, accum,
                strip_transpose(A), strip_transpose(B), replace_flag);
        }

    } // backend
} // GraphBLAS

#endif
