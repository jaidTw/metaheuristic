C++ = clang++
CPPFLAGS = -march=native -std=gnu++14 -Wall
debug:
	$(C++) $(CPPFLAGS) -Wextra -Werror -g -O0 PFSP.cpp -o PFSP
release:
	$(C++) $(CPPFLAGS) -O3 PFSP.cpp -o PFSP 
fast:
	$(C++) $(CPPFLAGS) -Ofast PFSP.cpp -o PFSP
