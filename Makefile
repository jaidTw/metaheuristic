C++ = clang++
CPPFLAGS = -march=native -std=gnu++14 -Wall
debug:
	$(C++) $(CPPFLAGS) -Wextra -Werror -g -O0 pfsp.cpp -o pfsp
release:
	$(C++) $(CPPFLAGS) -O3 pfsp.cpp -o pfsp
fast:
	$(C++) $(CPPFLAGS) -Ofast pfsp.cpp -o pfsp
