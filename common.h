#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

namespace MH {

    // All algorithms will use Solution to store the solution encoding and evaluation result (score).
    // Template parameter Encoding is the type of the encoding suppose to use.
    // this class is not supposed to be directly used by users.
    template <typename Encoding>
    struct Solution {
        friend bool operator<(Solution<Encoding> &a, Solution<Encoding> &b) {
            return a.score < b.score;
        }
        friend bool operator==(Solution<Encoding> &a, Solution<Encoding> &b) {
            return a.score == b.score && a.encoding == b.encoding;
        }
        // Solution encoding
        Encoding encoding;
        // Evaluation result
        double score;
        // Three types of constructor
        Solution();
        Solution(Encoding &);
        Solution(Encoding &, double);
    };

    // This collection will serve as the neighbours collection in trajectory algorithms,
    // and population in evolutionary algorithms
    template <typename Encoding>
    using SolCollection = std::vector<Solution<Encoding>>;

    // Replaces duplicate solutions with random ones.
    template <typename Encoding, typename Instance>
    inline void replaceDuplicates(SolCollection<Encoding> &solutions, Instance &instance) {
        static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
        for(size_t i = 0; i < solutions.size(); ++i) {
            for(size_t j = i + 1; j < solutions.size(); ++j) {
                if(solutions[i].score == solutions[j].score) {
                    if(solutions[i] == solutions[j]) {
                        std::iota(solutions[j].encoding.begin(), solutions[j].encoding.end(), 1);
                        std::shuffle(solutions[j].encoding.begin(), solutions[j].encoding.end(), eng);
                        solutions[j].score = instance.evaluate(solutions[j].encoding, instance.inf);
                    }
                }
            }
        }
    }
}

template <typename Encoding>
inline MH::Solution<Encoding>::Solution() : encoding(), score(0) {}

template <typename Encoding>
inline MH::Solution<Encoding>::Solution(Encoding &e) : encoding(e), score(0) {}

template <typename Encoding>
inline MH::Solution<Encoding>::Solution(Encoding &e, double s) : encoding(e), score(s){}