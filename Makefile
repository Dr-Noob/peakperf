CXX=gcc

CXXFLAGS1=-O2 -march=core-avx2 -fopenmp -DAVX_256 -DN_THREADS=8
CXXFLAGS2=-O2 -march=knl -fopenmp -DAVX_512 -DN_THREADS=256

MAIN=FLOPS.c
HASWELL=Haswell.c
KNL=Knl.c

OUTPUT1=256.out
OUTPUT2=512.out

$(OUTPUT1) $(OUTPUT2): Makefile $(MAIN) $(HASWELL) $(KNL)
	$(CXX) $(MAIN) $(HASWELL) $(CXXFLAGS1) -o $(OUTPUT1)
	$(CXX) $(MAIN) $(KNL) $(CXXFLAGS2) -o $(OUTPUT2)


clean:
	@rm $(OUTPUT1) $(OUTPUT2)
