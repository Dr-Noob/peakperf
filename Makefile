CXX=gcc

CXXFLAGS1=-O3 -march=core-avx2 -fopenmp
CXXFLAGS2=-O3 -march=knl -fopenmp

SOURCE1=intrinsics_multicore.c
SOURCE2=knl.c

OUTPUT1=multi.out
OUTPUT2=knl.out

all: Makefile $(SOURCE1) $(SOURCE2)
	$(CXX) $(SOURCE1) $(CXXFLAGS1) -o $(OUTPUT1)
	$(CXX) $(SOURCE2) $(CXXFLAGS2) -o $(OUTPUT2)


clean:
	@rm $(OUTPUT1) $(OUTPUT2)
