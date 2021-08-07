Requirements
==============

module load gcc/8.1.0
source /g/g90/velusamy/softwares/spack/share/spack/setup-env.sh
spack install gcc@9.3.0
spack install boost
spack install metall 
spack load gcc@9.3.0
spack load boost
spack load metall


git clone https://github.com/kaushikvelusamy/gbtl.git
cd gbtl

For Non Metall Case
====================

git checkout master

g++ -std=gnu++1z           -I./src/graphblas/detail           -I./src           -I./src/graphblas/platforms/sequential           src/demo/1_non_metall_bfs_demo.cpp          -o           1_non_metall_bfs_demo -O3

g++ -std=gnu++1z           -I./src/graphblas/detail           -I./src           -I./src/graphblas/platforms/sequential           src/demo/2_non_metall_tc_demo.cpp          -o           2_non_metall_tc_demo -O3

g++ -std=gnu++1z           -I./src/graphblas/detail           -I./src           -I./src/graphblas/platforms/sequential           src/demo/3_non_metall_pr_demo.cpp         -o           3_non_metall_pr_demo -O3



./1_non_metall_bfs_demo src/demo/gc-dataset/1_as20000102.txt
./2_non_metall_tc_demo src/demo/gc-dataset/1_as20000102.txt
./3_non_metall_pr_demo src/demo/gc-dataset/1_as20000102.txt 
