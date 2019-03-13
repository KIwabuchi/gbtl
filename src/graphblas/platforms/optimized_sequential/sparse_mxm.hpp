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
#include "sparse_mxm_AB.hpp"
#include "sparse_mxm_ATB.hpp"
#include "sparse_mxm_ABT.hpp"
#include "sparse_mxm_ATBT.hpp"
#include "LilSparseMatrix.hpp"


//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
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
                for (IndexType col_idx = 0; col_idx < ncol_B; ++col_idx)
                {
                    typename BMatrixT::ColType B_col(B.getCol(col_idx));
                    if (B_col.empty()) continue;

                    // create one rows of the result at a time
                    for (IndexType row_idx = 0; row_idx < nrow_A; ++row_idx)
                    {
                        typename AMatrixT::RowType A_row(A.getRow(row_idx));
                        if (A_row.empty()) continue;

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
