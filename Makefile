debug:
	c++ PFSP.cpp -o PFSP -Wall -Wextra -Werror -std=gnu++14 -g -Og
release:
	c++ PFSP.cpp -o PFSP -Wall -march=native -std=gnu++14 -O3
fast:
	c++ PFSP.cpp -o PFSP -Wall -march=native -std=gnu++14 -Ofast
