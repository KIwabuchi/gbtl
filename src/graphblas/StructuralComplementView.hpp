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

#include <cstddef>
#include <graphblas/Matrix.hpp>

//****************************************************************************
//****************************************************************************

namespace GraphBLAS
{
    //************************************************************************
    template<typename MatrixT>
    class MatrixStructuralComplementView
    {
    public:
        typedef bool  ScalarType;

        MatrixStructuralComplementView(MatrixT const &mat)
            : m_mat(mat)
        {
        }

        IndexType nrows() const { return m_mat.nrows(); }
        IndexType ncols() const { return m_mat.ncols(); }
        IndexType nvals() const
        {
            return (m_mat.nrows()*m_mat.ncols() - m_mat.nvals());
        }

        void printInfo(std::ostream &os) const
        {
            os << "MatrixStructuralComplementView of: ";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixStructuralComplementView const &mat)
        {
            os << "MatrixStructuralComplementView of: ";
            os << mat.m_mat;
            return os;
        }

        MatrixT const &m_mat;
    };

    //************************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structural_complement_v<ViewT>, int> = 0>
    decltype(auto)
    get_internal_matrix(ViewT const &view)
    {
        return MatrixStructuralComplementView(get_internal_matrix(view.m_mat));
    }


    //************************************************************************
    //************************************************************************

    //************************************************************************
    template<typename VectorT>
    class VectorStructuralComplementView
    {
    public:
        typedef bool  ScalarType;

        VectorStructuralComplementView(VectorT const &vec)
            : m_vec(vec)
        {
        }

        IndexType size()  const { return m_vec.size(); }
        IndexType nvals() const { return m_vec.size() - m_vec.nvals(); }

        void printInfo(std::ostream &os) const
        {
            os << "VectorStructuralComplementView of: ";
            m_vec.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        VectorStructuralComplementView const &vec)
        {
            os << "VectorStructuralComplementView of: ";
            os << vec.m_vec;
            return os;
        }

        VectorT const &m_vec;
    };

    //************************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structural_complement_v<ViewT>, int> = 0>
    decltype(auto)
    get_internal_vector(ViewT const &view)
    {
        return VectorStructuralComplementView(get_internal_vector(view.m_vec));
    }
}
