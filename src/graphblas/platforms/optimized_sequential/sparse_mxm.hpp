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
            GRB_LOG_VERBOSE("C := (A*B)");
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
            GRB_LOG_VERBOSE("C := C + (A*B)");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B)");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B)");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B)");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B)");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B)");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B)");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B)");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B)");
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
            GRB_LOG_VERBOSE("C := (A*B')");
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
            GRB_LOG_VERBOSE("C := C + (A*B')");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B')");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B')");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B')");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B')");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B')");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B')");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A*B')");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A*B')");
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
            GRB_LOG_VERBOSE("C := (A'*B)");
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
            GRB_LOG_VERBOSE("C := C + (A'*B)");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B)");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B)");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B)");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B)");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B)");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B)");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B)");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B)");
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
            GRB_LOG_VERBOSE("C := (A'*B')");
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
            GRB_LOG_VERBOSE("C := C + (A'*B')");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B')");
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
            GRB_LOG_VERBOSE("C<M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B')");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B')");
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
            GRB_LOG_VERBOSE("C<struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                      << " := (C + A'*B')");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B')");
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
            GRB_LOG_VERBOSE("C<!M" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B')");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (A'*B')");
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
            GRB_LOG_VERBOSE("C<!struct(M)" << ((outp == REPLACE) ? ",z>" : ">")
                            << " := (C + A'*B')");
            sparse_mxm_CompMask_Accum_ATBT(C, M_view.m_mat, true, accum, op,
                                           strip_transpose(A), strip_transpose(B), outp);
        }

    } // backend
} // GraphBLAS
