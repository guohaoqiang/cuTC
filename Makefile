
# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-102
# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart


ifdef ICC
CXX = icpc
WARN_FLAGS = -O3 -ipo -g -Wall -wd981 -wd383 -wd2259 -Werror # -Weffc++
else
CXX = g++
WARN_FLAGS = -O3 -g -Wall -Wextra -Wabi -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -Wstrict-null-sentinel -Woverloaded-virtual -Wshadow -Wcast-align -Wpointer-arith -Wwrite-strings -Wundef -Wredundant-decls -lglog -lgflags -lpthread# -Werror # -Weffc++
endif
NVXX = nvcc
NVXXFLAGS = -std=c++11 -lcutensor
NVXX_LINK_FLAGS =  -L./libcutensor/lib/10.2/ -I./libcutensor/include
#CXXFLAGS = -std=c++11 -O2 -lglogs -lgflags
#LINK_FLAGS = libgurobi_c++.a -L/home/ghq/gurobi912/linux64/lib -lgurobi91_light


SRC = ./src
BASE_SRC = ${SRC}/cutensor
CUTC_SRC = ${SRC}/kernels
INC = ./include
OBJ = ./obj

CC_SOURCE = $(wildcard ${SRC}/*.cc)
kernel_base_SOURCE = $(wildcard ${BASE_SRC}/*.cu)
kernel_cutc_SOURCE = $(wildcard ${CUTC_SRC}/*.cu)
CC_OBJECTS = $(patsubst %.cc,$(OBJ)/%.cc.o,$(notdir ${CC_SOURCE}))
kernel_base_OBJECTS = $(patsubst %.cu,$(OBJ)/%.cu.o,$(notdir ${kernel_base_SOURCE}))
kernel_cutc_OBJECTS = $(patsubst %.cu,$(OBJ)/%.cu.o,$(notdir ${kernel_cutc_SOURCE}))
#OBJECTS = main.o graphdata.o OptionParser.o acc.o pec.o analysis.o 

test:
	@echo ${BASE_SRC}
	@echo ${CC_OBJECTS}
	@echo ${kernel_base_OBJECTS}
	@echo ${kernel_base_SOURCE}

#BIN1 = tc
#BIN2 = cutensor
all:cutc mycutensor

#tc: $(CC_OBJECTS) $(kernel_cutc_OBJECTS)
#	$(CXX) -o $@ $^ $(WARN_FLAGS)

cutc: $(CC_OBJECTS) $(kernel_cutc_OBJECTS)
	$(CXX) -o $@ $^ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(NVXX_LINK_FLAGS) $(NVXXFLAGS) $(WARN_FLAGS) 

cutensor: $(CC_OBJECTS) $(kernel_base_OBJECTS)
	$(CXX) -o $@ $^ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(NVXX_LINK_FLAGS) $(NVXXFLAGS) $(WARN_FLAGS) 

#$(OBJ)/cutensor.cu.o: $(kernel_base_SOURCE) $(INC)/mycutensor.cuh
#	$(NVXX) -c $< -o $@ $(NVXXFLAGS) $(NVXX_LINK_FLAGS)
$(OBJ)/%.cc.o: $(SRC)/%.cc $(INC)/%.h
	$(CXX) -c $< -o $@ $(WARN_FLAGS) $(CXXFLAGS)

$(OBJ)/%.cu.o: $(BASE_SRC)/%.cu $(INC)/%.cuh
	$(NVXX) -c $< -o $@ $(NVXXFLAGS) $(NVXX_LINK_FLAGS)

$(OBJ)/%.cu.o: $(CUTC_SRC)/%.cu $(INC)/%.cuh
	$(NVXX) -c -G -g $< -o $@ $(NVXXFLAGS) $(NVXX_LINK_FLAGS)


.PHONY: clean

clean:
	find ${OBJ} -name *.o -exec rm -r {} \;
	rm -f $(BIN)
