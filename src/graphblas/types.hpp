/*
 * GraphBLAS Template Library, Version 3.0
 *
 * Copyright 2020, Carnegie Mellon University, Battelle Memorial Institute, and
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
 * RIGHTS.
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
 * 2. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/2.0.0/LICENSE) Copyright 2018 Carnegie
 * Mellon University, Battelle Memorial Institute, and Authors. DM18-0559
 *
 * DM20-0442
 */

#pragma once

#include <cstdint>
#include <exception>
#include <vector>
#include <iostream>

namespace GraphBLAS
{
    using IndexType = uint64_t;  /// @todo Consider template param for index type
    using IndexArrayType = std::vector<IndexType>;

    //**************************************************************************
    // When an operation uses a mask this controls what happens to non-masked
    // elements in the resulting container:
    //    MERGE   -> leave as is,
    //    REPLACE -> clear (annihilate) element.
    enum OutputControlEnum
    {
        MERGE = 0,
        REPLACE = 1
    };

    //**************************************************************************
    struct NoAccumulate
    {
        // It doesn't really matter what the type is, it never gets executed.
        template <typename D1=bool, typename D2=bool, typename D3=bool>
        inline D3 operator()(D1 lhs, D2 rhs) const { return true; }
    };

    //**************************************************************************
    class NoMask
    {
    public:
        friend std::ostream &operator<<(std::ostream             &os,
                                        NoMask          const    &mask)
        {
            os << "No mask";
            return os;
        }

        friend inline NoMask const &get_internal_matrix(NoMask const &mask)
        {
            return mask;
        }

        friend inline NoMask const &get_internal_vector(NoMask const &mask)
        {
            return mask;
        }
    };

    //**************************************************************************
    template<typename ScalarT, typename... TagsT> class Vector;

    template <class>
    inline constexpr bool is_vector_v = false;

    template <class ScalarT, class... Tags>
    inline constexpr bool is_vector_v<Vector<ScalarT, Tags...>> = true;

    //************************************************************************
    template<typename ScalarT, typename... TagsT> class Matrix;

    template <class>
    inline constexpr bool is_matrix_v = false;

    template <class ScalarT, class... Tags>
    inline constexpr bool is_matrix_v<Matrix<ScalarT, Tags...>> = true;

    //************************************************************************
    template<class VectorT> class VectorComplementView;
    template<class VectorT> class VectorStructureView;
    template<class VectorT> class VectorStructuralComplementView;

    template<class MatrixT> class TransposeView;
    template<class MatrixT> class MatrixComplementView;
    template<class MatrixT> class MatrixStructureView;
    template<class MatrixT> class MatrixStructuralComplementView;

    template <class MatrixT>
    inline constexpr bool is_matrix_v<TransposeView<MatrixT>> = true;

    //************************************************************************
    template <class>
    inline constexpr bool is_complement_v = false;

    template <class MatrixT>
    inline constexpr bool is_complement_v<MatrixComplementView<MatrixT>> = true;

    template <class VectorT>
    inline constexpr bool is_complement_v<VectorComplementView<VectorT>> = true;


    template <class>
    inline constexpr bool is_structure_v = false;

    template <class MatrixT>
    inline constexpr bool is_structure_v<MatrixStructureView<MatrixT>> = true;

    template <class VectorT>
    inline constexpr bool is_structure_v<VectorStructureView<VectorT>> = true;


    template <class>
    inline constexpr bool is_structural_complement_v = false;

    template <class MatrixT>
    inline constexpr bool is_structural_complement_v<
        MatrixStructuralComplementView<MatrixT>> = true;

    template <class VectorT>
    inline constexpr bool is_structural_complement_v<
        VectorStructuralComplementView<VectorT>> = true;


    template <class>
    inline constexpr bool is_transpose_v = false;

    template <class MatrixT>
    inline constexpr bool is_transpose_v<TransposeView<MatrixT>> = true;
}

namespace std
{
    /// @todo It seems that unit tests can't find this!
    // inline std::ostream &
    // operator<<(std::ostream &os, const std::vector<long unsigned int> vec)
    // {
    //     bool first = true;
    //     for (auto it = vec.begin(); it != vec.end(); ++it)
    //     {
    //         os << (first ? "" : ",") << *it;
    //         first = false;
    //     }
    //     return os;
    // }

} // namespace std
