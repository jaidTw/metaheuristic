C++ = g++
CPPFLAGS = -march=corei7-avx -std=gnu++14 -Wall
XOVER = -DUSE_OP
TRAJ = -DUSE_II_FI
POSTFIX = iifi
debug:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -Wextra -Werror -g -O0 pfsp.cpp -o pfsp_$(POSTFIX)
release:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -O3 pfsp.cpp -o pfsp_$(POSTFIX)
fast:
	$(C++) $(CPPFLAGS) $(XOVER) $(TRAJ) -Ofast pfsp.cpp -o pfsp_$(POSTFIX)

