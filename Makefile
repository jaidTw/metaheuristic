C++ = g++
CPPFLAGS = -march=corei7-avx -std=gnu++14 -Wall
<<<<<<< HEAD
=======
XOVER = -DUSE_OP
TRAJ = -DUSE_II_FI
POSTFIX = iifi_1k3h_calc
>>>>>>> 82b535cccb572f6b8b7cdc43b2ef7b765f773109
debug:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -Wextra -Wall -g -Og pfsp.cpp -o pfsp
release:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -O3 pfsp.cpp -o pfsp
fast:
	$(C++) $(CPPFLAGS) -Ofast pfsp.cpp -o pfsp
fastp:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -Ofast pfsp.cpp -o pfsp
