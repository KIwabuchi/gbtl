Requirements
==============

module load gcc/8.1.0
source /g/g90/velusamy/softwares/spack/share/spack/setup-env.sh
spack install gcc@9.3.0
spack install boost
spack install metall 
spack load gcc@9.3.0
spack load metall
spack load boost


git clone https://github.com/kaushikvelusamy/gbtl.git
cd gbtl

For Non Metall Case
====================

git checkout master

g++ -std=gnu++1z           -I./src/graphblas/detail           -I./src           -I./src/graphblas/platforms/sequential           src/demo/nvmw_non_metall_demo.cpp           -o           nvmw_non_metall_demo -O3

./nvmw_non_metall_demo src/demo/ca-AstroPh_adj.tsv
 

For Metall Case (Just check the datastore location in nvmw-metall.cpp) DRAM/NVM/Optane
================================================================================

git checkout metall-gbtl	
git submodule update --init

Delete previous datastore files
Goto CZTB2 and then ssh altus or skylake

				Machine Name		Datastore Location
With Metall DRAMB 	LLNL-Flash			/tmp/velusamy/datastore/
With Metall Optane	LLNL-Skylake		/mnt/pmem/pm0/datastore
With Metall nvm		LLNL-altus			/mnt/ssd/datastore



 

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_construction.cpp   -o  m_metall_construction.exe   -lstdc++fs   -O3


g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_algo1_bfs.cpp   -o  m_metall_algo1_bfs.exe   -lstdc++fs   -O3


g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_algo2_tc.cpp   -o  m_metall_algo2_tc.exe   -lstdc++fs   -O3






./m_metall_construction.exe src/demo/gc-datasets/1_facebook_combined_adj.tsv

 ./m_metall_algo1_bfs.exe

./m_metall_algo1_tc.exe
 





Remove last column on snap dataset cat 4_ca-AstroPh_adj.tsv | awk '{ print $1 "\t" $2}' > a4_ca-AstroPh_adj.tsv
