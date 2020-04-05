/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
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
 * Implementations of sparse vxm for the sequential (CPU) backend.
 */

#ifndef GB_SEQUENTIAL_SPARSE_VXM_HPP
#define GB_SEQUENTIAL_SPARSE_VXM_HPP

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
        /// Implementation of 4.3.2 vxm: Vector-Matrix multiply
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        inline void vxm(WVectorT          &w,
                        MaskT       const &mask,
                        AccumT             accum,
                        SemiringT          op,
                        UVectorT    const &u,
                        AMatrixT    const &A,
                        OutputControlEnum  outp)
        {
            // =================================================================
            // Do the basic dot-product work with the semi-ring.
            typedef typename SemiringT::result_type TScalarType;
            typedef typename AMatrixT::ScalarType AScalarType;
            typedef std::vector<std::tuple<IndexType,AScalarType> > AColType;

            std::vector<std::tuple<IndexType, TScalarType> > t;

            if ((A.nvals() > 0) && (u.nvals() > 0))
            {
                auto u_contents(u.getContents());
                for (IndexType col_idx = 0; col_idx < w.size(); ++col_idx)
                {
                    AColType const &A_col(A.getCol(col_idx));

                    if (!A_col.empty())
                    {
                        TScalarType t_val;
                        if (dot(t_val, u_contents, A_col, op))
                        {
                            t.push_back(std::make_tuple(col_idx, t_val));
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

    } // backend
} // GraphBLAS

#endif
