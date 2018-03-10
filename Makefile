CXX=gcc
CXXFLAGS=-O3 -march=core-avx2 -fopenmp

SOURCE=intrinsics_multicore.c
OUTPUT=multi.out

multi: Makefile $(SOURCE)
	$(CXX) $(SOURCE) $(CXXFLAGS) -o $(OUTPUT)

clean:
	@rm $(OUTPUT)
