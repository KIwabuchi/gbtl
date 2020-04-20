#include <iostream>

#include <graphblas/graphblas.hpp>

namespace GraphBLAS
{
    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class MatrixComplementView
    {
    public:
        typedef ScalarT ScalarType;
        typedef Matrix<ScalarT, TagsT...>  FrontendType;
        typedef typename FrontendType::BackendType  BackendType;

        MatrixComplementView(Matrix<ScalarT, TagsT...> const &mat)
            : m_mat(mat)
        {
        }

        void printInfo(std::ostream &os) const
        {
            os << "Frontend MatrixComplementView of:";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixComplementView const &mat)
        {
            os << "Frontend MatrixComplementView of:";
            mat.printInfo(os);
            return os;
        }

        Matrix<ScalarT, TagsT...> const &m_mat;

    };

    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class MatrixStructureView
    {
    public:
        typedef ScalarT ScalarType;
        typedef Matrix<ScalarT, TagsT...> FrontendType;
        typedef typename FrontendType::BackendType BackendType;

        MatrixStructureView(Matrix<ScalarT, TagsT...> const &mat)
            : m_mat(mat)
        {
        }

        void printInfo(std::ostream &os) const
        {
            os << "Frontend MatrixStructureView of:";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(std::ostream               &os,
                                        MatrixStructureView const &mat)
        {
            os << "Frontend MatrixStructureView of:";
            mat.printInfo(os);
            return os;
        }

        Matrix<ScalarT, TagsT...> const &m_mat;
    };

    //************************************************************************
    template<typename ScalarT, typename... TagsT>
    class MatrixStructuralComplementView
    {
    public:
        typedef ScalarT ScalarType;
        typedef Matrix<ScalarT, TagsT...> FrontendType;
        typedef typename FrontendType::BackendType BackendType;

        MatrixStructuralComplementView(Matrix<ScalarT, TagsT...> const &mat)
            : m_mat(mat)
        {
        }

        void printInfo(std::ostream &os) const
        {
            os << "Frontend MatrixStructuralComplementView of:";
            m_mat.printInfo(os);
        }

        friend std::ostream &operator<<(
            std::ostream               &os,
            MatrixStructuralComplementView const &mat)
        {
            os << "Frontend MatrixStructuralComplementView of:";
            mat.printInfo(os);
            return os;
        }

        FrontendType const &m_mat;
    };


    //************************************************************************
    template <class>
    inline constexpr bool is_complement_v = false;

    template <class T, class... Tags>
    inline constexpr bool is_complement_v<MatrixComplementView<T, Tags...>> = true;

    template <class>
    inline constexpr bool is_structure_v = false;

    template <class T, class... Tags>
    inline constexpr bool is_structure_v<MatrixStructureView<T, Tags...>> = true;


    template <class>
    inline constexpr bool is_structural_complement_v = false;

    template <class T, class... Tags>
    inline constexpr bool is_structural_complement_v<MatrixStructuralComplementView<T, Tags...>> = true;


}

//****************************************************************************
//****************************************************************************

namespace GraphBLAS
{
    namespace backend
    {
        //********************************************************************
        template <class BackendMatrixT>
        class MatrixComplementView2
        {
        public:
            typedef BackendMatrixT BackendType;

            MatrixComplementView2(BackendMatrixT const &matrix)
                : m_mat(matrix)
            {
            }

            friend std::ostream &operator<<(
                std::ostream               &os,
                MatrixComplementView2 const &mat)
            {
                os << "backend::MatrixComplementView2 of:";
                mat.m_mat.printInfo(os);
                return os;
            }

            BackendMatrixT const &m_mat;
        };

        //********************************************************************
        template <class BackendMatrixT>
        class MatrixStructureView2
        {
        public:
            typedef BackendMatrixT BackendType;

            MatrixStructureView2(BackendMatrixT const &matrix)
                : m_mat(matrix)
            {
            }

            friend std::ostream &operator<<(
                std::ostream               &os,
                MatrixStructureView2 const &mat)
            {
                os << "backend::MatrixSructureView of:";
                mat.m_mat.printInfo(os);
                return os;
            }

            BackendMatrixT const &m_mat;
        };

        //********************************************************************
        template <class BackendMatrixT>
        class MatrixStructuralComplementView
        {
        public:
            typedef BackendMatrixT BackendType;

            MatrixStructuralComplementView(BackendMatrixT const &matrix)
                : m_mat(matrix)
            {
            }

            friend std::ostream &operator<<(
                std::ostream                         &os,
                MatrixStructuralComplementView const &mat)
            {
                os << "backend::MatrixSructuralComplementView of:";
                mat.m_mat.printInfo(os);
                return os;
            }

            BackendMatrixT const &m_mat;
        };


        //********************************************************************
        template<typename CMatrixT,
                 typename MaskT,
                 typename AMatrixT,
                 typename BMatrixT>
        inline void grb_op(CMatrixT         &C,
                           MaskT      const &Mask,
                           AMatrixT   const &A,
                           BMatrixT   const &B)
        {
            std::cout << "grb_op - backend - begin\n";
            std::cout << "backend Mask in : " << Mask << std::endl;
            std::cout << "grb_op - backend - end\n";
        }
    }
}

namespace GraphBLAS
{
    //********************************************************************
    // template <class MatrixT,
    //           typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    // typename MatrixT::BackendType &transfer_matrix(MatrixT &matrix)
    // {
    //     return matrix.m_mat;
    // }

    // //********************************************************************
    // template <class MatrixT,
    //           typename std::enable_if_t<is_matrix_v<MatrixT>, int> = 0>
    // typename MatrixT::BackendType const &transfer_matrix(MatrixT const &matrix)
    // {
    //     return matrix.m_mat;
    // }

    //********************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_complement_v<ViewT>, int> = 0>
    auto
    transfer_matrix(ViewT const &view)
    {
        return backend::MatrixComplementView2(view.m_mat);
    }

    //********************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structure_v<ViewT>, int> = 0>
    auto
    transfer_matrix(ViewT const &view)
    {
        return backend::MatrixStructureView2(view.m_mat);
    }

    //********************************************************************
    template <class ViewT,
              typename std::enable_if_t<is_structural_complement_v<ViewT>, int> = 0>
    auto
    transfer_matrix(ViewT const &view)
    {
        return backend::MatrixStructuralComplementView(view.m_mat);
    }

    //************************************************************************
    // 4.3.1: Matrix-matrix multiply
    template<typename CMatrixT,
             typename MaskT,
             typename AMatrixT,
             typename BMatrixT>
    inline void grb_op(CMatrixT         &C,
                       MaskT      const &Mask,
                       AMatrixT   const &A,
                       BMatrixT   const &B)
    {
        std::cout << "grb_op - frontend - begin\n";
        std::cout << "frontend Mask in : " << Mask << std::endl;

        backend::grb_op(transfer_matrix(C),
                        transfer_matrix(Mask),
                        transfer_matrix(A),
                        transfer_matrix(B));

        std::cout << "grb_op - frontend - end\n";
    }

    //************************************************************************
    // Views
    //************************************************************************

    /**
     * @brief  Return a view that structures the structure of a matrix.
     * @param[in]  Mask  The matrix to structure
     *
     */
    template<typename ScalarT, typename... TagsT>
    MatrixComplementView<ScalarT, TagsT...> complement2(
        Matrix<ScalarT, TagsT...> const &Mask)
    {
        return MatrixComplementView<ScalarT, TagsT...>(Mask);
    }

    /**
     * @brief  Return a view that complements the structure of a matrix.
     * @param[in]  Mask  The matrix to complement
     *
     */
    template<typename ScalarT, typename... TagsT>
    MatrixStructureView<ScalarT, TagsT...> structure2(
        Matrix<ScalarT, TagsT...> const &Mask)
    {
        return MatrixStructureView<ScalarT, TagsT...>(Mask);
    }

    /**
     * @brief  Return a view that structures the structure of a matrix.
     * @param[in]  Mask  The matrix to structure
     *
     */
    template<typename ScalarT, typename... TagsT>
    MatrixStructuralComplementView<ScalarT, TagsT...> complement2(
        MatrixStructureView<ScalarT, TagsT...> const &structure_view)
    {
        return MatrixStructuralComplementView<ScalarT, TagsT...>(
            structure_view.m_mat);
    }
}


//****************************************************************************
int main(int argc, char **argv)
{
    std::vector<std::vector<double>> Atmp = {{1, 1, 0},
                                             {2, 2, 0},
                                             {0, 2, 3}};
    std::vector<std::vector<int>> Mtmp = {{1, 0, 0},
                                          {0, 1, 0},
                                          {0, 1, 1}};
    GraphBLAS::Matrix<double, GraphBLAS::DirectedMatrixTag> AA(Atmp, 0.0);
    GraphBLAS::Matrix<int>  C(3,3), A(3,3), B(3,3);
    GraphBLAS::Matrix<int>  M(Mtmp, 0);

    GraphBLAS::grb_op(C, M,
                      A, B);
    GraphBLAS::grb_op(C, GraphBLAS::structure2(M),
                      A, B);
    GraphBLAS::grb_op(C, GraphBLAS::complement2(M),
                      A, B);
    GraphBLAS::grb_op(C, GraphBLAS::complement2(GraphBLAS::structure2(M)),
                      A, B);
    //GraphBLAS::grb_op(C, GraphBLAS::structure2(GraphBLAS::complement2(M)),
    //                  A, B); // does not compile
}
