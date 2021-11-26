ifdef ICC
CXX = icpc
WARN_FLAGS = -O3 -ipo -g -Wall -wd981 -wd383 -wd2259 -Werror # -Weffc++
else
CXX = g++
WARN_FLAGS = -O3 -g -Wall -Wextra -Wabi -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -Wstrict-null-sentinel -Woverloaded-virtual -Wshadow -Wcast-align -Wpointer-arith -Wwrite-strings -Wundef -Wredundant-decls -lglog -lgflags -lpthread# -Werror # -Weffc++
endif
NVXX = nvcc
#CXXFLAGS = -std=c++11 -O2 -lglogs -lgflags
#LINK_FLAGS = libgurobi_c++.a -L/home/ghq/gurobi912/linux64/lib -lgurobi91_light


SRC = ./src
INC = ./include
OBJ = ./obj

CC_SOURCE = $(wildcard ${SRC}/*.cc)
kernel_SOURCE = $(wildcard ${SRC}/*.cu)
CC_OBJECTS = $(patsubst %.cc,$(OBJ)/%.cc.o,$(notdir ${CC_SOURCE}))
kernel_OBJECTS = $(patsubst %.cu,$(OBJ)/%.cu.o,$(notdir ${kernel_SOURCE}))
#OBJECTS = main.o graphdata.o OptionParser.o acc.o pec.o analysis.o 


BIN = tc
$(BIN): $(CC_OBJECTS) $(kernel_OBJECTS)
	$(CXX) -o $@ $^ $(WARN_FLAGS)

$(OBJ)/%.cc.o: $(SRC)/%.cc $(INC)/%.h
	$(CXX) -c $< -o $@ $(WARN_FLAGS) $(CXXFLAGS)

$(OBJ)/%.cu.o: $(SRC)/%.cu $(INC)/%.h
	$(NVXX) -c $< -o $@



.PHONY: clean

clean:
	find ${OBJ} -name *.o -exec rm -r {} \;
	rm -f $(BIN)
