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



spack find -p boost

/home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj

Run /home/perma/dropcache - bofore construction and algorithm

For altus system 
------------------

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_construction.cpp   -o  m_metall_construction.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_algo1_bfs.cpp   -o  m_metall_algo1_bfs.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_algo2_tc.cpp   -o  m_metall_algo2_tc.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread ./src/demo/m_metall_algo3_pr.cpp   -o  m_metall_algo3_pr.exe   -lstdc++fs   -O3

For Flash system
------------------

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./g/g90/velusamy/softwares/spack/opt/spack/linux-rhel7-haswell/gcc-8.1.0/boost-1.73.0-fxe7xvtjsbexcqdl6genvt7kxgvbsogs -pthread ./src/demo/m_metall_construction.cpp   -o  m_metall_construction.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./g/g90/velusamy/softwares/spack/opt/spack/linux-rhel7-haswell/gcc-8.1.0/boost-1.73.0-fxe7xvtjsbexcqdl6genvt7kxgvbsogs -pthread ./src/demo/m_metall_algo1_bfs.cpp   -o  m_metall_algo1_bfs.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./g/g90/velusamy/softwares/spack/opt/spack/linux-rhel7-haswell/gcc-8.1.0/boost-1.73.0-fxe7xvtjsbexcqdl6genvt7kxgvbsogs -pthread ./src/demo/m_metall_algo2_tc.cpp   -o  m_metall_algo2_tc.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./g/g90/velusamy/softwares/spack/opt/spack/linux-rhel7-haswell/gcc-8.1.0/boost-1.73.0-fxe7xvtjsbexcqdl6genvt7kxgvbsogs -pthread ./src/demo/m_metall_algo3_pr.cpp   -o  m_metall_algo3_pr.exe   -lstdc++fs   -O3


For annonymous map
------------------

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread -DMETALL_USE_ANONYMOUS_NEW_MAP -DMETALL_INITIAL_SEGMENT_SIZE=$((2**26)) ./src/demo/m_metall_construction.cpp   -o  m_metall_construction.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread -DMETALL_USE_ANONYMOUS_NEW_MAP -DMETALL_INITIAL_SEGMENT_SIZE=$((2**26)) ./src/demo/m_metall_algo1_bfs.cpp   -o  m_metall_algo1_bfs.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread -DMETALL_USE_ANONYMOUS_NEW_MAP -DMETALL_INITIAL_SEGMENT_SIZE=$((2**26)) ./src/demo/m_metall_algo2_tc.cpp   -o  m_metall_algo2_tc.exe   -lstdc++fs   -O3

g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/ -I./home/velusamy/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/boost-1.73.0-wu3z2ipyy4yzsje7afep3xkenjeeq4fj -pthread -DMETALL_USE_ANONYMOUS_NEW_MAP -DMETALL_INITIAL_SEGMENT_SIZE=$((2**26)) ./src/demo/m_metall_algo3_pr.cpp   -o  m_metall_algo3_pr.exe   -lstdc++fs   -O3



./m_metall_construction.exe src/demo/gc-datasets/1_facebook_combined_adj.tsv

 ./m_metall_algo1_bfs.exe NUM_NODES SRC DEST 

./m_metall_algo2_tc.exe
 
./m_metall_algo3_pr.exe NUM_NODES





To Remove last column on snap dataset 

cat 4_ca-AstroPh_adj.tsv | awk '{ print $1 "\t" $2}' > a4_ca-AstroPh_adj.tsv

To Convert comma separated files to tab separated files. 

sed 's/,/\t/g' report.csv > report.tsv
