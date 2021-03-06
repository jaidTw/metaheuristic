C++ = g++
CPPFLAGS = -march=corei7-avx -std=gnu++14 -Wall
debug:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -Wextra -Wall -g -Og pfsp.cpp -o pfsp
release:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -O3 pfsp.cpp -o pfsp
fast:
	$(C++) $(CPPFLAGS) -Ofast pfsp.cpp -o pfsp
fastp:
	$(C++) $(CPPFLAGS) -DEVOL_PLOT -Ofast pfsp.cpp -o pfsp
