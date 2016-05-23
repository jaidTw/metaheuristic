#pragma once

#include "common.h"
#include <cstdint>
#include <random>
#include <chrono>
#include <algorithm>
#include <valarray>
#include <iostream>
#include <tuple>

namespace MH {
    namespace Evolutionary {
        template <typename SelectionPolicy, typename CrossoverPolicy>
        class DE;

        template<typename>
        struct is_vector { static bool const value = false; };
        template<typename T>
        struct is_vector<std::vector<T>> { static bool const value = true; };
        template<typename>
        struct is_DE { static bool const value = false; };
        template<typename S, typename C>
        struct is_DE<DE<S, C>> { static bool const value = true; };


        template <typename Encoding>
        struct Instance {
            // if Algorithm is not DE or Encoding is not std::vector
            template <typename Algorithm>
            Solution<Encoding> evolution(Algorithm &, std::vector<Encoding> &,
                typename std::enable_if<!is_DE<Algorithm>::value || !is_vector<Encoding>::value>::type* = nullptr);
            // if Algorithm is DE and Encoding is std::vector
            template <typename Algorithm>
            Solution<Encoding> evolution(Algorithm &, std::vector<Encoding> &,
                typename std::enable_if<is_DE<Algorithm>::value && is_vector<Encoding>::value>::type* = nullptr);

            uint64_t generationLimit;
            void *inf;
            // Neighbourhood generator: accepts an encoding and returns a vector of neighbourhood encodings.
            std::vector<Encoding> (*neighbourhood)(Encoding &);
            double (*evaluate)(Encoding &, void *);
            void (*mutate)(Encoding &, double);
        };

        // selection policies
        class Tournament {
        public:
            size_t size;
        };

        // crossover policies
        class OP {}; // one-point crossover
        class OX {}; // order crossover
        class PMX {}; // partially-mapped crossover
        // TODO: class SJOX {}; // similar job order crossover
        // TODO: class LOX {}; // linear order crossover
        // TODO: class CX {}; // cycle crossover

        // mutation policies
        class Shift {};

        // Encoding of DE is restricted to vector of real numbers (float | double | long double).
        template <typename SelectionPolicy, typename CrossoverPolicy>
        class DE {
        public:
            double crossover_rate;
            double current_factor;
            double scaling_factor;
            uint8_t num_of_diff_vectors;

            template <typename Encoding>
            void initialise(Instance<Encoding> &, std::vector<Encoding> &);
            template <typename Encoding> 
            Encoding mate(Encoding &, SolCollection<Encoding> &);
        private:
            template <typename Encoding>
            static Encoding mutation(std::vector<Encoding> &, SolCollection<Encoding> &, double, uint8_t);
        };

        template <typename FP>
        struct _DE_INF_WRAPPER {
            double (*original_evaluate)(std::vector<FP> &, void *);
            void *original_inf;
        };

        // DE selection policies
        class DE_Random {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding select(std::vector<Encoding> &, SolCollection<Encoding> &, double);
        };
        class DE_Best {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding select(std::vector<Encoding> &, SolCollection<Encoding> &, double);
        };
        class DE_CurrentToRandom {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding select(std::vector<Encoding> &, SolCollection<Encoding> &, double);
        };
        class DE_CurrentToBest {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding select(std::vector<Encoding> &, SolCollection<Encoding> &, double);
        };

        // DE crossover policies
        class DE_None {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding crossover(Encoding &, Encoding &, double);
        };
        class DE_Binomial {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding crossover(Encoding &, Encoding &, double);
        };
        class DE_Exponential {
            template <typename S, typename C>
            friend class DE;
            template <typename Encoding>
            static Encoding crossover(Encoding &, Encoding &, double);
        };

        template <typename Encoding, typename Selection, typename Crossover, typename LocalSearch, typename LSInstance>
        struct MA {
            void initialise(Instance<Encoding> &, std::vector<Encoding> &);
            bool _offspringAreParents;
            bool elitism;
            bool removeDuplicates;
            double mutationProbability; // Should be between 0 and 1, inclusive.
            LocalSearch localSearch;
            LSInstance lsInstance;
            SolCollection<Encoding> offspring;
            Selection selectionPolicy;
            Crossover crossoverPolicy;
        };


        template <typename FP>
        class _valarray_eq_binder {
        public:
            _valarray_eq_binder(std::valarray<FP> &array) : _array(array) {}
            _valarray_eq_binder() {}
            bool operator()(std::valarray<FP> &array) {
                return std::equal(std::begin(array), std::end(array), std::begin(_array));
            }
            void set(std::valarray<FP> &array) { _array = array; }
        private:
            std::valarray<FP> _array;
        };

        // aliases
        template <typename... Args>
        using DifferentialEvolution = DE<Args...>;
        template <typename... Args>
        using MemeticAlgorithm = MA<Args...>;

        // Function definitions

        template <typename Encoding>
        SolCollection<Encoding> initialisePopulation(Instance<Encoding> &, std::vector<Encoding> &);

        template <typename Encoding, typename... DEArgs>
        void generate(Instance<Encoding> &, SolCollection<Encoding> &, DE<DEArgs...> &);

        template <typename Encoding, typename... MAArgs>
        void generate(Instance<Encoding> &, SolCollection<Encoding> &, MA<Encoding, MAArgs...> &);

        template <typename Encoding, typename... MAArgs>
        inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
        mate(Instance<Encoding> &, SolCollection<Encoding> &, MA<Encoding, MAArgs...> &);

        template <typename Encoding>
        inline size_t mateSelect(SolCollection<Encoding> &, Tournament &);

        template <typename Encoding>
        inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
        crossover(Instance<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, OP &);

        template <typename Encoding>
        inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
        crossover(Instance<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, OX &);

        template <typename Encoding>
        inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
        crossover(Instance<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, PMX &);

        template <typename FP>
        double _DE_EVALUATE_WRAPPER(std::valarray<FP> &, void *);
    }
}

template <typename Encoding> template <typename Algorithm>
MH::Solution<Encoding>
MH::Evolutionary::Instance<Encoding>::evolution(Algorithm &algorithm, std::vector<Encoding> &init,
    typename std::enable_if<!is_DE<Algorithm>::value || !is_vector<Encoding>::value>::type*) {

    algorithm.initialise(*this, init);
    auto population = MH::Evolutionary::initialisePopulation(*this, init);

    for(auto generationCount = 0UL; generationCount < generationLimit; ++generationCount) {
        MH::Evolutionary::generate(*this, population, algorithm);

#ifdef EVOL_PLOT
        std::cout << generationCount + 1 << " " << std::min_element(population.begin(), population.end())->score << std::endl;
#endif

    }
    auto min = *std::min_element(population.begin(), population.end());
    return min;
}


template <typename Encoding> template <typename Algorithm>
MH::Solution<Encoding>
MH::Evolutionary::Instance<Encoding>::evolution(Algorithm &de, std::vector<Encoding> &init,
    typename std::enable_if<is_DE<Algorithm>::value && is_vector<Encoding>::value>::type*) {

    typedef typename Encoding::value_type FP;
    // this wrapper wrap original evaluate function pointer and inf to new instance's inf
    _DE_INF_WRAPPER<FP> wrapper;
    wrapper.original_evaluate = evaluate;
    wrapper.original_inf = inf;

    // instance set to valarray type
    auto WrapInstance = Instance<std::valarray<FP>>();
    // this evaluation function wrapper will restore original evaluator from inf
    WrapInstance.evaluate = _DE_EVALUATE_WRAPPER;
    WrapInstance.generationLimit = generationLimit;
    WrapInstance.inf = reinterpret_cast<void *>(&wrapper);

    // convert real vector to valarray
    std::vector<std::valarray<FP>> valarray_init(init.size());
    std::transform(init.begin(), init.end(), valarray_init.begin(),
                   [&](auto &s) {
                       return std::valarray<FP>(s.data(), s.size());
                   });
    auto result = WrapInstance.evolution(de, valarray_init);

    // convert vallarray back to real vector
    Encoding vec_result(std::begin(result.encoding), std::end(result.encoding));
    return Solution<std::vector<FP>>(vec_result, result.score);
}

template <typename S, typename C> template <typename Encoding>
inline void
MH::Evolutionary::DE<S, C>::initialise(Instance<Encoding> &, std::vector<Encoding> &) {
}

template <typename Encoding, typename S, typename C, typename LSA, typename LSI>
inline void
MH::Evolutionary::MA<Encoding, S, C, LSA, LSI>::initialise(Instance<Encoding> &, std::vector<Encoding> &init) {
    _offspringAreParents = false;

    offspring.resize(init.size());
    for(auto &elem : offspring) {
        elem.encoding.resize(init.front().size());
    }
}

template <typename Encoding>
inline MH::SolCollection<Encoding>
MH::Evolutionary::initialisePopulation(Instance<Encoding> &instance, std::vector<Encoding> &init) {

    MH::SolCollection<Encoding> population(init.size());
    std::transform(init.begin(), init.end(), population.begin(),
                   [&](auto &s) {
                       return MH::Solution<Encoding>(s, instance.evaluate(s, instance.inf));
                   });
    return population;
}

template <typename Encoding, typename... MAArgs>
inline void
MH::Evolutionary::generate(Instance<Encoding> &instance, SolCollection<Encoding> &population, MA<Encoding, MAArgs...> &ma) {

    auto &thePopulation = (ma._offspringAreParents) ? ma.offspring : population;
    auto &theOffspring = (ma._offspringAreParents) ? population : ma.offspring;

    for(size_t i = 0; i < population.size(); i += 2) {
        Solution<Encoding> offspring1, offspring2;

        std::tie(offspring1, offspring2) = MH::Evolutionary::mate(instance, thePopulation, ma);

        // local search
        theOffspring[i] = ma.lsInstance.search(ma.localSearch, offspring1.encoding);
        theOffspring[i + 1] = ma.lsInstance.search(ma.localSearch, offspring2.encoding);
    }

    // elitism
    if(ma.elitism) {
        auto min = std::min_element(thePopulation.begin(), thePopulation.end());
        auto max = std::max_element(theOffspring.begin(), theOffspring.end());
        *max = *min;
    }
    
    // Remove duplicates to avoid early convergence to a local optimum.
    if(ma.removeDuplicates) {
        replaceDuplicates(theOffspring, instance);
    }

    // Let ma.offspring and population take turns acting as the parents.
    ma._offspringAreParents = !ma._offspringAreParents;
}

template <typename Encoding, typename... MAArgs>
inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
MH::Evolutionary::mate(Instance<Encoding> &instance, SolCollection<Encoding> &population, MA<Encoding, MAArgs...> &ma) {

    auto parent1 = MH::Evolutionary::mateSelect(population, ma.selectionPolicy);
    auto parent2 = MH::Evolutionary::mateSelect(population, ma.selectionPolicy);

    while(parent2 == parent1) {
        parent2 = MH::Evolutionary::mateSelect(population, ma.selectionPolicy);
    }
    return MH::Evolutionary::crossover(instance, population[parent1], population[parent2], ma.mutationProbability, ma.crossoverPolicy);
}

template <typename Encoding>
inline size_t
MH::Evolutionary::mateSelect(SolCollection<Encoding> &population,
                             Tournament &tournament) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    
    std::vector<size_t> contestants;
    for(size_t i = 0; i < tournament.size; ++i) {
        contestants.push_back(eng() % population.size());
    }

    size_t winner = 0;
    for(size_t i = 1; i < contestants.size(); ++i) {
        if(population[contestants[i]] < population[contestants[winner]]) {
            winner = i;
        }
    }

    return contestants[winner];
}

// OP: encoding is limited to job indices.
template <typename Encoding>
inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            Solution<Encoding> &parent1,
                            Solution<Encoding> &parent2,
                            double mutationProbability,
                            Evolutionary::OP &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

    auto offspring1(parent1);
    auto offspring2(parent2);

    size_t size = parent1.encoding.size();
    std::vector<bool> knockout1(size, false), knockout2(size, false);
    size_t crossoverPoint = eng() % size;
    for(size_t i = 0; i < crossoverPoint; ++i) {
        offspring1.encoding[i] = parent1.encoding[i];
        knockout1[parent1.encoding[i] - 1] = true;
        offspring2.encoding[i] = parent2.encoding[i];
        knockout2[parent2.encoding[i] - 1] = true;
    }
    size_t off1Index = crossoverPoint;
    size_t off2Index = crossoverPoint;
    for(size_t i = 0; i < size; ++i) {
        if(!knockout1[parent2.encoding[i] - 1]) offspring1.encoding[off1Index++] = parent2.encoding[i];
        if(!knockout2[parent1.encoding[i] - 1]) offspring2.encoding[off2Index++] = parent1.encoding[i];
    }
    instance.mutate(offspring1.encoding, mutationProbability);
    instance.mutate(offspring2.encoding, mutationProbability);
    offspring1.score = instance.evaluate(offspring1.encoding, instance.inf);
    offspring2.score = instance.evaluate(offspring2.encoding, instance.inf);
    return std::tie(offspring1, offspring2);
}

// OX: encoding is limited to job indices.
template <typename Encoding>
inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            Solution<Encoding> &parent1,
                            Solution<Encoding> &parent2,
                            double mutationProbability,
                            Evolutionary::OX &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

    auto offspring1(parent1);
    auto offspring2(parent2);

    size_t size = parent1.encoding.size();
    std::vector<bool> knockout1(size, false), knockout2(size, false);
    size_t crossoverPointA = eng() % (size - 1);
    size_t crossoverPointB = eng() % (size - crossoverPointA - 1) + crossoverPointA + 1;
    for(size_t i = crossoverPointA; i < crossoverPointB; ++i) {
        offspring1.encoding[i] = parent1.encoding[i];
        knockout1[parent1.encoding[i] - 1] = true;
        offspring2.encoding[i] = parent2.encoding[i];
        knockout2[parent2.encoding[i] - 1] = true;
    }
    size_t off1Index = 0;
    size_t off2Index = 0;
    for(size_t i = 0; i < size; ++i) {
        if(off1Index == crossoverPointA) {
            off1Index = crossoverPointB;
        }
        if(!knockout1[parent2.encoding[i] - 1]) {
            offspring1.encoding[off1Index++] = parent2.encoding[i];
        }
        if(off2Index == crossoverPointA) {
            off2Index = crossoverPointB;
        }
        if(!knockout2[parent1.encoding[i] - 1]) {
            offspring2.encoding[off2Index++] = parent1.encoding[i];
        }
    }
    instance.mutate(offspring1.encoding, mutationProbability);
    instance.mutate(offspring2.encoding, mutationProbability);
    offspring1.score = instance.evaluate(offspring1.encoding, instance.inf);
    offspring2.score = instance.evaluate(offspring2.encoding, instance.inf);
    return std::tie(offspring1, offspring2);
}

// PMX: encoding is limited to job indices.
template<typename Encoding>
size_t PMXHelper(MH::Solution<Encoding> &parent1,
                 MH::Solution<Encoding> &parent2,
                 MH::Solution<Encoding> &offspring,
                 size_t crossoverPointA,
                 size_t crossoverPointB,
                 size_t searchIndex,
                 size_t valueIndex) {
    size_t i;
    for(i = 0; i < parent1.encoding.size(); ++i) {
        if(parent2.encoding[i] == parent1.encoding[searchIndex]) {
            break;
        }
    }
    if(i >= crossoverPointA && i < crossoverPointB) {
        i = PMXHelper(parent1, parent2, offspring, crossoverPointA, crossoverPointB, i, valueIndex);
    }
    else {
        offspring.encoding[i] = parent2.encoding[valueIndex];
    }
    return i;
}

template <typename Encoding>
inline std::tuple<MH::Solution<Encoding>, MH::Solution<Encoding>>
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            Solution<Encoding> &parent1,
                            Solution<Encoding> &parent2,
                            double mutationProbability,
                            Evolutionary::PMX &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

    auto offspring1(parent1);
    auto offspring2(parent2);

    size_t size = parent1.encoding.size();

    std::vector<bool> knockout1(size, false), knockout2(size, false);

    std::fill(offspring1.encoding.begin(), offspring1.encoding.end(), 0);
    std::fill(offspring2.encoding.begin(), offspring2.encoding.end(), 0);

    size_t crossoverPointA = eng() % (size - 1);
    size_t crossoverPointB = eng() % (size - crossoverPointA - 1) + crossoverPointA + 1;
    for(size_t i = crossoverPointA; i < crossoverPointB; ++i) {
        offspring1.encoding[i] = parent1.encoding[i];
        knockout1[parent1.encoding[i] - 1] = true;
        offspring2.encoding[i] = parent2.encoding[i];
        knockout2[parent2.encoding[i] - 1] = true;
    }
    for(size_t i = crossoverPointA; i < crossoverPointB; ++i) {
        if(!knockout1[parent2.encoding[i] - 1]) {
            PMXHelper(parent1, parent2, offspring1, crossoverPointA, crossoverPointB, i, i);
            knockout1[parent2.encoding[i] - 1] = true;
        }
        if(!knockout2[parent1.encoding[i] - 1]) {
            PMXHelper(parent2, parent1, offspring2, crossoverPointA, crossoverPointB, i, i);
            knockout2[parent1.encoding[i] - 1] = true;
        }
    }
    for(size_t i = 0; i < size; ++i) {
        if(offspring1.encoding[i] == 0) {
            offspring1.encoding[i] = parent2.encoding[i];
        }
        if(offspring2.encoding[i] == 0) {
            offspring2.encoding[i] = parent1.encoding[i];
        }
    }
    instance.mutate(offspring1.encoding, mutationProbability);
    instance.mutate(offspring2.encoding, mutationProbability);
    offspring1.score = instance.evaluate(offspring1.encoding, instance.inf);
    offspring2.score = instance.evaluate(offspring2.encoding, instance.inf);
    return std::tie(offspring1, offspring2);
}

template <typename Encoding, typename... DEArgs>
inline void
MH::Evolutionary::generate(Instance<Encoding> &instance,
                           std::vector<Solution<Encoding>> &population,
                           MH::Evolutionary::DE<DEArgs...> &de) {
    for(auto i = 0UL; i < population.size(); ++i) {
        auto target_vec = population[i].encoding;
        auto trial_vec = de.mate(target_vec, population);
        // environment selection
        auto trial_score = instance.evaluate(trial_vec, instance.inf);
        if(trial_score < population[i].score) {
            population[i] = MH::Solution<Encoding>(trial_vec, trial_score);
        }
    }
}


template <typename SelectionPolicy, typename CrossoverPolicy> template <typename Encoding> 
Encoding
MH::Evolutionary::DE<SelectionPolicy, CrossoverPolicy>::mate(Encoding &target_vec, SolCollection<Encoding> &population) {
    std::vector<Encoding> selectionPool;
    selectionPool.push_back(target_vec);
    auto mutant_vec = SelectionPolicy::select(selectionPool, population, current_factor);
    mutant_vec += mutation(selectionPool, population, scaling_factor, num_of_diff_vectors);
    auto trial_vec = CrossoverPolicy::crossover(target_vec, mutant_vec, crossover_rate);
    return trial_vec;
}


template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_Random::select(std::vector<Encoding> &selectionPool, SolCollection<Encoding> &population, double) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<size_t> uniform(0, population.size() - 1);

    selectionPool.push_back(population[uniform(eng)].encoding);
    return selectionPool.back();
}

template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_Best::select(std::vector<Encoding> &selectionPool, SolCollection<Encoding> &population, double) {
    selectionPool.push_back(std::min_element(population.begin(), population.end())->encoding);
    return selectionPool.back();
}

template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_CurrentToRandom::select(std::vector<Encoding> &selectionPool, SolCollection<Encoding> &population, double factor) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<size_t> uniform(0, population.size() - 1);

    selectionPool.push_back(population[uniform(eng)].encoding);
    return selectionPool.front() + factor * (selectionPool.back() - selectionPool.front());
}

template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_CurrentToBest::select(std::vector<Encoding> &selectionPool, SolCollection<Encoding> &population, double factor) {
    selectionPool.push_back(std::min_element(population.begin(), population.end())->encoding);
    return selectionPool.front() + factor * (selectionPool.back() - selectionPool.front());
}


template <typename S, typename C> template <typename Encoding>
inline Encoding
MH::Evolutionary::DE<S, C>::mutation(std::vector<Encoding> &selectionPool,
                                     SolCollection<Encoding> &population,
                                     double factor, uint8_t diff_vecs) {
    // random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<size_t> uniform(0, population.size() - 1);

    auto sol1 = selectionPool.front();
    auto sol2 = selectionPool.front();
    auto mutant_vec = Encoding(sol1.size());

    // Find 2 * n mutual different solutions to generate n vectos.
    for(auto i = 0U; i < diff_vecs; ++i) {
        // valarray equality functor as an predicate for std::find_if()
        _valarray_eq_binder<double> valarr_eq;

        // Select the solution if is not already in selection pool.
        do {
            auto rand_idx = uniform(eng);
            sol1 = population[rand_idx].encoding;
            valarr_eq.set(sol1);
        } while(std::find_if(selectionPool.begin(), selectionPool.end(), valarr_eq) != selectionPool.end());
        selectionPool.push_back(sol1);
        do {
            auto rand_idx = uniform(eng);
            sol2 = population[rand_idx].encoding;
            valarr_eq.set(sol2);
        } while(std::find_if(selectionPool.begin(), selectionPool.end(), valarr_eq) != selectionPool.end());
        selectionPool.push_back(sol2);
        mutant_vec += sol2 - sol1;
    }
    return factor * mutant_vec;
}

// No crossover; do nothing.
template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_None::crossover(Encoding &, Encoding &mutant_vec, double) {
    return mutant_vec;
}

// DE binomial crossover
template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_Binomial::crossover(Encoding &target_vec, Encoding &mutant_vec, double crossover_rate) {
    // random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> uniform_r;
    static std::uniform_int_distribution<size_t> uniform_i(0, mutant_vec.size());

    Encoding trial_vec(target_vec.size());

    // Combine two vectors into trial vector.
    std::transform(std::begin(target_vec), std::end(target_vec),
                   std::begin(mutant_vec), std::begin(trial_vec),
                   [&](auto &target, auto &mutant) {
                   // The uniform real generator will result in an infinite loop in g++ 6.1.1.
                   // The result is abnormal in g++ 5.3.1.
                   // Works in clang 3.7.1.
                       return uniform_r(eng) < crossover_rate ? mutant : target;
                   });

    // Guarantee at least one random field is inherited from mutant vector.
    auto pos = uniform_i(eng);
    trial_vec[pos] = mutant_vec[pos];
    return trial_vec;
}

// DE exponential crossover
template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_Exponential::crossover(Encoding &target_vec, Encoding &mutant_vec, double crossover_rate) {
    // Random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> uniform_r;
    static std::uniform_int_distribution<size_t> uniform_i(0, mutant_vec.size());

    // Generate a random crossover starting position.
    auto pos = uniform_i(eng);

    auto trial_vec(target_vec);
    for(auto i = 0UL; i < target_vec.size(); ++i) {
        trial_vec[pos] = mutant_vec[pos];
        pos = (pos + 1) % target_vec.size();
 
        if(uniform_r(eng) >= crossover_rate) {
            break;
        }
    }
    return trial_vec;
}

template <typename FP>
inline double
MH::Evolutionary::_DE_EVALUATE_WRAPPER(std::valarray<FP> &sol, void *inf) {
    // Convert the valarray back to a vector.
    std::vector<FP> vsol(std::begin(sol), std::end(sol));

    // Restore the original evaluator and inf from the wrapper.
    auto wrapper = *reinterpret_cast<MH::Evolutionary::_DE_INF_WRAPPER<FP> *>(inf);

    // Now we can use the original evaluator.
    return wrapper.original_evaluate(vsol, wrapper.original_inf);
}
