#pragma once

#include "common.h"
#include <cstdint>
#include <cmath>
#include <vector>
#include <deque>
#include <limits>
#include <iostream>

namespace MH {
    namespace Trajectory {
        // A trajectory instance object specifies the generation limit, the neighbourhood generator, and the evaluator.
        // There is also an optional "inf" field to provide additional information to evaluation function.
        template <typename Encoding>
        struct Instance {
            uint64_t generationLimit;
            // Neighbourhood generator: accepts an encoding and returns a vector of neighbourhood encodings.
            std::vector<Encoding> (*neighbourhood)(Encoding &);
            // Evaluator: accepts an encoding and additional information from "inf" as (void *)
            // and returns a real number, which is suppose to be minimised.
            double (*evaluate)(Encoding &, void *);
            // optional information provided to evaluator
            void *inf;
        };

        template <typename Policy>
        class II {
            template <typename E, typename A>
            friend Solution<E> search(Instance<E> &, A &, E &);
            typedef Policy policy;
        public:
            II();
        private:
            template <typename Encoding>
            void initialise(Instance<Encoding> &, Encoding &);
            template <typename Encoding>
            Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &);
            double _prevScore;
            policy _policy;
        };

        // II search policies
        class II_BestImproving {
            template <typename Policy>
            friend class II;
            template <typename Encoding>
            static Solution<Encoding>& select(Solution<Encoding> &, SolCollection<Encoding> &);
        };
        class II_FirstImproving {
            template <typename Policy>
            friend class II;
            template <typename Encoding>
            static Solution<Encoding>& select(Solution<Encoding> &, SolCollection<Encoding> &);

        };
        // TODO : class II_Stochastic { };

        class SA {
            template <typename E, typename A>
            friend Solution<E> search(Instance<E> &, A &, E &);
        public:
            double init_temperature;
            uint64_t epoch_length;
            double (*cooling)(double);

        private:
            template <typename Encoding>
            void initialise(Instance<Encoding> &, Encoding &);
            template <typename Encoding>
            Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &);
            template <typename Encoding>
            Solution<Encoding>& selectHelper(Solution<Encoding> &, SolCollection<Encoding> &);
            double _temperature;
            uint64_t _epoch_count;
        };

        template <typename Encoding, typename TraitType>
        class TS {
            template <typename E, typename A>
            friend Solution<E> search(Instance<E> &, A &, E &);
        public:
            uint8_t length;
            // the trait function accept an encoding and transform it into traits to store in tabu list.
            TraitType (*trait)(Encoding&, void *);
        private:
            void initialise(Instance<Encoding> &, Encoding &);
            Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &);
            std::deque<TraitType> _queue;
        };

        // These basic searches is aim to compare with others
        // TODO  : struct RS {};
        // TODO  : struct BFS {};
        // TODO  : struct DFS {};

        // aliases
        template <typename Arg>
        using IterativeImprovement = II<Arg>;
        using SimulatedAnnealing = SA;
        template <typename... Args>
        using TabuSearch = TS<Args...>;

        // Function declarations
        template <typename Encoding, typename Algorithm>
        Solution<Encoding> search(Instance<Encoding> &, Algorithm &, Encoding &);
        template <typename Encoding>
        Solution<Encoding>& select_SA(double, Solution<Encoding> &, SolCollection<Encoding> &);
    }
}

template <typename Policy>
MH::Trajectory::II<Policy>::II() : _prevScore(std::numeric_limits<double>::infinity()) {}

// The main search framework for trajectory-based algorithms
template <typename Encoding, typename Algorithm>
MH::Solution<Encoding>
MH::Trajectory::search(MH::Trajectory::Instance<Encoding> &instance,
                       Algorithm &algorithm,
                       Encoding &init) {

    algorithm.initialise(instance, init);
    auto current = Solution<Encoding>(init, instance.evaluate(init, instance.inf));
    auto min = current;

    for(uint64_t generationCount = 0;
        generationCount < instance.generationLimit;
        ++generationCount) {

        auto neighbours_encoding = instance.neighbourhood(current.encoding);
        MH::SolCollection<Encoding> neighbours(neighbours_encoding.size());
        // Evaluate each encoding, and store in the solution vector.
        std::transform(neighbours_encoding.begin(), neighbours_encoding.end(),
                       neighbours.begin(),
                       [&](auto &e) {
                           return Solution<Encoding>(e, instance.evaluate(e, instance.inf));
                       });

        // Each algorithm differs as to its selection mechanism.
        current = algorithm.select(instance, current, neighbours);
        if(current < min) {
            min = current;
        }
        std::cout << generationCount + 1 << " " << current.score << std::endl;
    }
    return min;
}

template <typename Policy> template <typename Encoding>
inline void
MH::Trajectory::II<Policy>::initialise(Instance<Encoding> &instance, Encoding &) {
}

template <typename Encoding>
inline void
MH::Trajectory::SA::initialise(Instance<Encoding> &, Encoding &) {
    _temperature = init_temperature;
}


template <typename Encoding, typename TraitType>
inline void
MH::Trajectory::TS<Encoding, TraitType>::initialise(Instance<Encoding> &instance, Encoding &init) {
    _queue.resize(length);
    std::fill(_queue.begin(), _queue.end(), trait(init, instance.inf));
}

// Best improving II: select the minimum among the neighbours.
template <typename Encoding>
inline MH::Solution<Encoding>&
MH::Trajectory::II_BestImproving::select(Solution<Encoding> &current,
                                         SolCollection<Encoding> &neighbours) {

    auto &min = *std::min_element(neighbours.begin(), neighbours.end()); 
    return (min < current) ? min : current;
}

// First improving II: select the first solution that is better than current one during the iteration.
template <typename Encoding>
inline MH::Solution<Encoding>&
MH::Trajectory::II_FirstImproving::select(Solution<Encoding> &current,
                                          SolCollection<Encoding> &neighbours) {
    for(auto &neighbour : neighbours) {
        if(neighbour < current) {
            return neighbour;
        }
    }
    return current;
}

// II selection: do inner selection based on II policy.
template <typename Policy> template <typename Encoding>
inline MH::Solution<Encoding>&
MH::Trajectory::II<Policy>::select(Instance<Encoding> &instance,
                                   Solution<Encoding> &current,
                                   SolCollection<Encoding> &neighbours) {
    auto &result = Policy::select(current, neighbours);

    // Stop the search if at a local optimum.
    if(result.score >= _prevScore) {
        instance.generationLimit = 0;
    }
    else {
        _prevScore = result.score;
    }
    return result;
}

// SA selection: call select_SA and handle the cooling schedule.
template <typename Encoding>
inline MH::Solution<Encoding>&
MH::Trajectory::SA::select(Instance<Encoding> &instance,
                           Solution<Encoding> &current,
                           SolCollection<Encoding> &neighbours) {
    auto &result = selectHelper(current, neighbours);
    ++_epoch_count;
    if(_epoch_count == epoch_length) {
        _temperature = cooling(_temperature);
        _epoch_count = 0;
    }
    return result;
}

// TS selection: compare neighbours with the tabu list; choose the minimum not in the list
// and replace the oldest solution with the new solution.
template <typename Encoding, typename TraitType>
inline MH::Solution<Encoding>&
MH::Trajectory::TS<Encoding, TraitType>::select(Instance<Encoding> &instance,
                                                Solution<Encoding> &,
                                                SolCollection<Encoding> &neighbours) {
    auto &min = neighbours.front();
    for(auto &neighbour : neighbours) {
        if(std::find(_queue.begin(), _queue.end(),
                     trait(neighbour.encoding, instance.inf)) == _queue.end() &&
           neighbour < min) {
            min = neighbour;
        }
    }
    _queue.pop_front();
    _queue.push_back(trait(min.encoding, instance.inf));
    return min;
}

// SA selection
template <typename Encoding>
inline MH::Solution<Encoding>&
MH::Trajectory::SA::selectHelper(Solution<Encoding> &current,
                                 SolCollection<Encoding> &neighbours) {
    // random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> uniform;

    for(auto &neighbour : neighbours) {
        if((neighbour < current) ||
           (exp((current.score - neighbour.score)/ _temperature) > uniform(eng)) ) {
            return neighbour;
        }
    }
    return current;
}