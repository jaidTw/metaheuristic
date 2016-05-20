C++ = g++
CPPFLAGS = -march=corei7-avx -std=gnu++14 -Wall
XOVER = -DUSE_OP
TRAJ = -DUSE_SA
debug:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -Wextra -Werror -g -O0 pfsp.cpp -o pfsp
release:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -O3 pfsp.cpp -o pfsp
fast:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -Ofast pfsp.cpp -o pfsp
parallel:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -O3 pfsp.cpp -o pfsp -D_GLIBCXX_PARALLEL -fopenmp

