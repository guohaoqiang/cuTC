ifdef ICC
CXX = icpc
WARN_FLAGS = -O3 -ipo -g -Wall -wd981 -wd383 -wd2259 -Werror # -Weffc++
else
CXX = g++
WARN_FLAGS = -O3 -g -Wall -Wextra -Wabi -Wctor-dtor-privacy -Wnon-virtual-dtor -Wreorder -Wstrict-null-sentinel -Woverloaded-virtual -Wshadow -Wcast-align -Wpointer-arith -Wwrite-strings -Wundef -Wredundant-decls -lglog -lgflags -lpthread -lgurobi91_light# -Werror # -Weffc++
endif
LINK_FLAGS = libgurobi_c++.a -L/home/ghq/gurobi912/linux64/lib -lgurobi91_light


SRC = ./src
INC = ./include
OBJ = ./obj

SOURCE = $(wildcard ${SRC}/*.cc)
BIN = sim
OBJECTS = $(patsubst %.cc,$(OBJ)/%.o,$(notdir ${SOURCE}))
#OBJECTS = main.o graphdata.o OptionParser.o acc.o pec.o analysis.o 


$(BIN): $(OBJECTS) 
	$(CXX) -o $@ $(OBJECTS) $(LINK_FLAGS) $(WARN_FLAGS)

$(OBJ)/%.o: $(SRC)/%.cc $(INC)/%.h
	$(CXX) $(WARN_FLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	find ${OBJ} -name *.o -exec rm -r {} \;
	rm -f $(BIN)
