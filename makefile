CXX=g++

INC_DIR = ./include
SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin

INCLUDE  += -I $(INC_DIR)
CFLAGS = -lglfw3 -lm -lGL -lGLU -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lSOIL -lIL -fpermissive
CPPFLAGS += -std=c++11

NAME = physsim

OBJS = $(OBJ_DIR)/Calc.o \
	$(OBJ_DIR)/Core.o \
	$(OBJ_DIR)/Entity.o \
	$(OBJ_DIR)/GUI.o \
	$(OBJ_DIR)/Init.o \
	$(OBJ_DIR)/Particle.o \
	$(OBJ_DIR)/TUI.o \
	$(OBJ_DIR)/UserInterface.o \
	$(OBJ_DIR)/Vector.o \

all: directories $(BIN_DIR)/$(NAME)

$(BIN_DIR)/$(NAME): $(OBJS)
	$(CXX) $(OBJS) -o $(BIN_DIR)/$(NAME) $(CFLAGS) $(CPPFLAGS) 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) $(CPPFLAGS)  $(INCLUDE) -c $< -o $@

.PHONY: directories
directories:
	mkdir -p $(OBJ_DIR)

clean::
	-rm $(OBJS) $(INC_DIR)/*.*~ $(SRC_DIR)/*.*~

.PHONY: clean
