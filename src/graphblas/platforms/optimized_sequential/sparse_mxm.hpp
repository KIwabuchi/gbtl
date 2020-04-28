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
                                 OutputControlEnum    outp)
        {
            // Dimension checks happen in front end
            IndexType nrow_A(A.nrows());
            IndexType ncol_B(B.ncols());

            typedef typename CMatrixT::ScalarType   CScalarType;
            typedef typename SemiringT::result_type TScalarType;
            typedef std::vector<std::tuple<IndexType,TScalarType> > TColType;

            // =================================================================
            // Do the basic dot-product work with the semiring.
            LilSparseMatrix<TScalarType> T(nrow_A, ncol_B);

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
                                TScalarType T_val;
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
                T.recomputeNvals();
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

            LilSparseMatrix<ZScalarType> Z(C.nrows(), C.ncols());

            ewise_or_opt_accum(Z, C, T, accum);

            GRB_LOG_VERBOSE("Z: " << Z);

            // =================================================================
            // Copy Z into the final output considering mask and replace/merge
            write_with_opt_mask(C, Z, M, outp);

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
                        OutputControlEnum     outp)
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
                        OutputControlEnum     outp)
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
                        OutputControlEnum     outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_AB(C, M, false, op, A, B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat              &C,
                        MMat      const   &M,
                        Accum     const   &accum,
                        SR                 op,
                        AMat      const   &A,
                        BMat      const   &B,
                        OutputControlEnum  outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_Mask_Accum_AB(C, M, false, accum, op, A, B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_AB(C, M_view.m_mat, true, op, A, B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_Mask_Accum_AB(C, M_view.m_mat, true, accum, op, A, B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_AB(C, M_view.m_mat, false,
                                           op, A, B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_CompMask_Accum_AB(C, M_view.m_mat, false, accum,
                                         op, A, B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_AB(C, M_view.m_mat, true,
                                           op, A, B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        AMat                       const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B)" << std::endl;
            sparse_mxm_CompMask_Accum_AB(C, M_view.m_mat, true, accum,
                                         op, A, B, outp);
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
                        OutputControlEnum          outp)
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
                        OutputControlEnum          outp)
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
                        OutputControlEnum          outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ABT(C, M, false, op,
                                        A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        AMat                const &A,
                        TransposeView<BMat> const &B,
                        OutputControlEnum          outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_Mask_Accum_ABT(C, M, false, accum, op,
                                      A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ABT(C, M_view.m_mat, true, op,
                                        A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                &C,
                        MatrixStructureView<MMat>   const  &M_view,
                        Accum                        const  &accum,
                        SR                                   op,
                        AMat                         const  &A,
                        TransposeView<BMat>          const  &B,
                        OutputControlEnum                    outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_Mask_Accum_ABT(C, M_view.m_mat, true, accum, op,
                                      A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ABT(C, M_view.m_mat, false, op,
                                            A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                &C,
                        MatrixComplementView<MMat>   const  &M_view,
                        Accum                        const  &accum,
                        SR                                   op,
                        AMat                         const  &A,
                        TransposeView<BMat>          const  &B,
                        OutputControlEnum                    outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ABT(C, M_view.m_mat, false, accum, op,
                                          A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        AMat                       const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ABT(C, M_view.m_mat, true, op,
                                            A, strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                                &C,
                        MatrixStructuralComplementView<MMat>   const  &M_view,
                        Accum                        const  &accum,
                        SR                                   op,
                        AMat                         const  &A,
                        TransposeView<BMat>          const  &B,
                        OutputControlEnum                    outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ABT(C, M_view.m_mat, true, accum, op,
                                          A, strip_transpose(B), outp);
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
                        OutputControlEnum            outp)
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
                        OutputControlEnum            outp)
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
                        OutputControlEnum            outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_ATB(C, M, false, op,
                                        strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        BMat                const &B,
                        OutputControlEnum          outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_Mask_Accum_ATB(C, M, false, accum, op,
                                      strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_Mask_NoAccum_ATB(C, M_view.m_mat, true, op,
                                        strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_Mask_Accum_ATB(C, M_view.m_mat, true, accum, op,
                                      strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATB(C, M_view.m_mat, false, op,
                                            strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_CompMask_Accum_ATB(C, M_view.m_mat, false, accum, op,
                                          strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B)" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATB(C, M_view.m_mat, true, op,
                                            strip_transpose(A), B, outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        BMat                       const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B)" << std::endl;
            sparse_mxm_CompMask_Accum_ATB(C, M_view.m_mat, true, accum, op,
                                          strip_transpose(A), B, outp);
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
                        OutputControlEnum          outp)
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
                        OutputControlEnum          outp)
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
                        OutputControlEnum          outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ATBT(C, M, false, op,
                                         strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                      &C,
                        MMat                const &M,
                        Accum               const &accum,
                        SR                         op,
                        TransposeView<AMat> const &A,
                        TransposeView<BMat> const &B,
                        OutputControlEnum          outp)
        {
            std::cout << "C<M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_Mask_Accum_ATBT(C, M, false, accum, op,
                                       strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_Mask_NoAccum_ATBT(C, M_view.m_mat, true, op,
                                         strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructureView<MMat>  const &M_view,
                        Accum                      const &accum,
                        SR                                 op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_Mask_Accum_ATBT(C, M_view.m_mat, true, accum, op,
                                       strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATBT(C, M_view.m_mat, false, op,
                                             strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                 op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ATBT(C, M_view.m_mat, false, accum, op,
                                           strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        NoAccumulate               const &,
                        SR                                op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (A'*B')" << std::endl;
            sparse_mxm_CompMask_NoAccum_ATBT(C, M_view.m_mat, true, op,
                                             strip_transpose(A), strip_transpose(B), outp);
        }

        template<class CMat, class MMat, class Accum, class SR, class AMat, class BMat>
        inline void mxm(CMat                             &C,
                        MatrixStructuralComplementView<MMat> const &M_view,
                        Accum                      const &accum,
                        SR                                 op,
                        TransposeView<AMat>        const &A,
                        TransposeView<BMat>        const &B,
                        OutputControlEnum                 outp)
        {
            std::cout << "C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B')" << std::endl;
            sparse_mxm_CompMask_Accum_ATBT(C, M_view.m_mat, true, accum, op,
                                           strip_transpose(A), strip_transpose(B), outp);
        }

    } // backend
} // GraphBLAS
