CXX=gcc
CXXFLAGS=-O3 -march=core-avx2 -fopenmp

SOURCE=intrinsics_multinucleo.c
OUTPUT=multi

multi: Makefile $(SOURCE)
	$(CXX) $(SOURCE) $(CXXFLAGS) -o $(OUTPUT)

clean:
	@rm $(OUTPUT)
