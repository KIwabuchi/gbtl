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
        /// Implementation of 4.3.1 mxm: Matrix-matrix multiply
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
            //Frontend checks the dimensions, but use C explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef typename BMatrixT::ScalarType BScalarType;
            typedef typename CMatrixT::ScalarType CScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarType> > CColType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TColType;

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            LilSparseMatrix<D3ScalarType> T(nrow_A, ncol_B);

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                // create a column of result at a time
                TColType T_col;
                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    typename BMatrixT::ColType B_col(B.getCol(col_idx));

                    if (!B_col.empty())
                    {
                        for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                        {
                            typename AMatrixT::RowType A_row(A.getRow(row_idx));
                            if (!A_row.empty())
                            {
                                D3ScalarType T_val;
                                if (dot(T_val, A_row, B_col, op))
                                {
                                    T_col.push_back(
                                            std::make_tuple(row_idx, T_val));
                                }
                            }
                        }
                        if (!T_col.empty())
                        {
                            T.setCol(col_idx, T_col);
                            T_col.clear();
                        }
                    }
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                D3ScalarType,
                typename AccumT::result_type>::type ZScalarType;
            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, M, replace_flag);

        } // mxm
#endif

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
            //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType nrow_B(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            if ((A.nvals() == 0) && (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================
            // Do the basic axpy work with the semiring.
            CRowType C_row;

            // Build this completely based on the semiring
            for (IndexType i = 0; i < nrow_A; ++i)
            {
                C_row.clear();
                for (auto const &Ai_elt : A[i])
                {
                    IndexType    k(std::get<0>(Ai_elt));
                    AScalarT  a_ik(std::get<1>(Ai_elt));

                    if (B[k].empty()) continue;

                    auto Ci_iter = C_row.begin();
                    for (auto const &Bk_elt : B[k])
                    {
                        IndexType    j(std::get<0>(Bk_elt));
                        BScalarT  b_kj(std::get<1>(Bk_elt));

                        // scan through C_row to find insert/merge point
                        IndexType j_C;
                        while ((Ci_iter != C_row.end()) &&
                               ((j_C = std::get<0>(*Ci_iter)) < j))
                        {
                            ++Ci_iter;
                        }

                        auto t_ij(semiring.mult(a_ik, b_kj));

                        if (Ci_iter == C_row.end())
                        {
                            C_row.push_back(
                                std::make_tuple(j, static_cast<CScalarT>(t_ij)));
                            Ci_iter = C_row.end();
                        }
                        else if (j_C == j)
                        {
                            std::get<1>(*Ci_iter) =
                                semiring.add(std::get<1>(*Ci_iter), t_ij);
                            ++Ci_iter;
                        }
                        else
                        {
                            Ci_iter = C_row.insert(
                                Ci_iter,
                                std::make_tuple(j, static_cast<CScalarT>(t_ij)));
                            ++Ci_iter;
                        }
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
        inline void sparse_mxm_NoMask_Accum_AB(
            LilSparseMatrix<CScalarT>       &C,
            AccumT                           accum,
            SemiringT                        semiring,
            LilSparseMatrix<AScalarT> const &A,
            LilSparseMatrix<BScalarT> const &B)
        {
            std::cout << "sparse_mxm_NoMask_Accum_AB not implemented yet.\n";
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
            std::cout << "sparse_mxm_Mask_NoAccum_AB not implemented yet.\n";
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
            std::cout << "sparse_mxm_Mask_Accum_AB not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_NoAccum_AB not implemented yet.\n";
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
            std::cout << "sparse_mxm_CompMask_Accum_AB not implemented yet.\n";
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
            //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType ncol_BT(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;

            if ((A.nvals() == 0) && (B.nvals() == 0))
            {
                C.clear();
                return;
            }

            // =================================================================
            // Do the basic dot-product work with the semiring.
            CRowType C_row;

            // Build this completely based on the semiring
            for (IndexType i = 0; i < nrow_A; ++i)
            {
                if (A[i].empty()) continue;

                // fill row i of T
                for (IndexType j = 0; j < ncol_BT; ++j)
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
                C_row.clear();
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
            //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType ncol_BT(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef typename AccumT::result_type ZScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;
            typedef std::vector<std::tuple<IndexType,ZScalarType> > ZRowType;

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            TRowType T_row;
            ZRowType Z_row;

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                for (IndexType i = 0; i < nrow_A; ++i)
                {
                    if (A[i].empty()) continue;

                    // fill row i of T
                    for (IndexType j = 0; j < ncol_BT; ++j)
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
                        ewise_or(Z_row, C[i], T_row, accum);
                        C.setRow(i, Z_row);
                        Z_row.clear();
                        T_row.clear();
                    }
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
             //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType ncol_BT(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            LilSparseMatrix<D3ScalarType> T(nrow_A, ncol_BT);
            TRowType T_row;

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                for (IndexType i = 0; i < nrow_A; ++i)
                {
                    auto A_row_i(A.getRow(i));
                    if (A_row_i.empty()) continue;

                    // fill row i of T
                    for (IndexType j = 0; j < ncol_BT; ++j)
                    {
                        auto B_row_j(B.getRow(j));
                        if (B_row_j.empty()) continue;

                        D3ScalarType t_ij;

                        // Perform the dot product
                        if (dot(t_ij, A_row_i, B_row_j, semiring))
                        {
                            T_row.push_back(std::make_tuple(j, t_ij));
                        }

                        if (!T_row.empty())
                        {
                            T.setRow(i, T_row);
                            T_row.clear();
                        }
                    }
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, T, M, replace_flag);

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
             //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType ncol_BT(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,CScalarT> > CRowType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            LilSparseMatrix<D3ScalarType> T(nrow_A, ncol_BT);
            TRowType T_row;

            // Build this completely based on the semiring
            if ((A.nvals() > 0) && (B.nvals() > 0))
            {
                for (IndexType i = 0; i < nrow_A; ++i)
                {
                    auto A_row_i(A.getRow(i));
                    if (A_row_i.empty()) continue;

                    // fill row i of T
                    for (IndexType j = 0; j < ncol_BT; ++j)
                    {
                        auto B_row_j(B.getRow(j));
                        if (B_row_j.empty()) continue;

                        D3ScalarType t_ij;

                        // Perform the dot product
                        if (dot(t_ij, A_row_i, B_row_j, semiring))
                        {
                            T_row.push_back(std::make_tuple(j, t_ij));
                        }

                        if (!T_row.empty())
                        {
                            T.setRow(i, T_row);
                            T_row.clear();
                        }
                    }
                }
            }

            GRB_LOG_VERBOSE("T: " << T);

            // =================================================================
            // Accumulate into Z
            typedef typename std::conditional<
                std::is_same<AccumT, NoAccumulate>::value,
                D3ScalarType,
                typename AccumT::result_type>::type ZScalarType;
            LilSparseMatrix<ZScalarType> Z(nrow_C, ncol_C);

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace
            write_with_opt_mask(C, Z, M, replace_flag);

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
            std::cout << "sparse_mxm_CompMask_NoAccum_ABT not implemented yet.";
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
            std::cout << "sparse_mxm_CompMask_Accum_ABT not implemented yet.";
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
            //Frontend checks the dimensions, but use C dimensions explicitly
            IndexType nrow_C(C.nrows());
            IndexType ncol_C(C.ncols());

            IndexType nrow_A(A.nrows());
            IndexType nrow_B(B.nrows());

            typedef typename SemiringT::result_type D3ScalarType;
            typedef std::vector<std::tuple<IndexType,D3ScalarType> > TRowType;

            C.clear();

            if ((A.nvals() == 0) && (B.nvals() == 0))
            {
                return;
            }

            // =================================================================
            // Do the basic axpy work with the semiring.
            TRowType Ci_tmp;

            for (IndexType k = 0; k < nrow_A; ++k)
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

                    C.mergeRow(i, Ci_tmp, AdditiveMonoidFromSemiring<SemiringT>(semiring));
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
            std::cout << "sparse_mxm_NoMask_Accum_AB not implemented yet.\n";
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
