#pragma once

#include <cstdint>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <algorithm>
#include <chrono>
#include <deque>
#include <valarray>

// Declarations
// All things in this library will be in MH namespace
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

        // The II algorithm class
        // II< II_BestImproving | II_FirstImproving | II_Stochastic >
        template <typename Strategy>
        struct II {
            II(size_t theGenerationLimit) :
            generationLimit(theGenerationLimit),
            prevScore(std::numeric_limits<double>::infinity()) {}

            size_t generationLimit;
            double score;
            double prevScore;
            Strategy _strategy;
        };

        // II search strategies
        struct II_BestImproving {};
        struct II_FirstImproving {};
        struct II_Stochastic {};

        // The SA algorithm class
        struct SA {
            double init_temperature;
            uint64_t epoch_length;
            double (*cooling)(double);
            double _temperature;
            uint64_t _epoch_count;
        };

        template <typename Encoding, typename TraitType>
        struct TS {
            uint8_t length;
            // the trait function accept an encoding and transform it into traits to store in tabu list.
            TraitType (*trait)(Encoding&, void *);
            std::deque<TraitType> _queue;
        };

        // These basic searches is aim to compare with others
        struct RS {};
        struct BFS {};
        struct DFS {};

        // aliases
        template <typename Arg>
        using IterativeImprovement = II<Arg>;
        using SimulatedAnnealing = SA;
        template <typename... Args>
        using TabuSearch = TS<Args...>;
        using RandomSearch = RS;
        using BreadthFirstSearch = BFS;
        using DepthFirstSearch = DFS;

        // Function declarations
        template <typename Encoding, typename Algorithm>
        Solution<Encoding> search(Instance<Encoding> &, Algorithm &, Encoding &);

        template <typename Encoding, typename Strategy>
        void initialise(Instance<Encoding> &, II<Strategy> &, Encoding &);

        template <typename Encoding>
        void initialise(Instance<Encoding> &, SA &, Encoding &);

        template <typename Encoding, typename TraitType>
        void initialise(Instance<Encoding> &, TS<Encoding, TraitType> &, Encoding &);

        template <typename Encoding>
        void initialise(Instance<Encoding> &, RS &, Encoding &);

        template <typename Encoding, typename Strategy>
        Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, II<Strategy> &);

        template <typename Encoding>
        Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, SA &);

        template <typename Encoding, typename Trait>
        Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, TS<Encoding, Trait> &);

        template <typename Encoding>
        Solution<Encoding>& select(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, RS &);

        template <typename Encoding>
        Solution<Encoding>& select_II(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, II_BestImproving &);

        template <typename Encoding>
        Solution<Encoding>& select_II(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, II_FirstImproving &);

        template <typename Encoding>
        Solution<Encoding>& select_II(Instance<Encoding> &, Solution<Encoding> &, SolCollection<Encoding> &, II_Stochastic &);

        template <typename Encoding>
        Solution<Encoding>& select_SA(double, Solution<Encoding> &, SolCollection<Encoding> &);
    }

    namespace Evolutionary {

        // selection strategies
        class Tournament {
        public:
            size_t size;
        };

        // crossover strategies
        class OP {}; // one-point crossover
        // TODO: class SJOX {}; // similar job order crossover
        class OX {}; // order crossover
        // TODO: class LOX {}; // linear order crossover
        class PMX {}; // partially-mapped crossover
        // TODO: class CX {}; // cycle crossover

        // mutation strategies
        class Shift {};

        // DE selection strategies
        class DE_Random {};
        class DE_Best {};
        class DE_CurrentToRandom {};
        class DE_CurrentToBest {};

        // DE crossover strategies
        class DE_None {};
        class DE_Binomial {};
        class DE_Exponential {};

        // Encoding of DE is restricted to vector of real numbers (float | double | long double).
        template <typename Selection, typename Crossover>
        struct DE {
            double crossover_rate;
            double current_factor;
            double scaling_factor;
            uint8_t num_of_diff_vectors;
            Selection _selection_strategy;
            Crossover _crossover_strategy;
        };

        template <typename Encoding, typename Selection, typename Crossover, typename LocalSearch, typename LSInstance>
        struct MA {
            MA(size_t, size_t, bool, bool, double, LocalSearch &, LSInstance &);
            bool _offspringAreParents;
            bool elitism;
            bool removeDuplicates;
            double mutationProbability; // Should be between 0 and 1, inclusive.
            LocalSearch localSearch;
            LSInstance lsInstance;
            SolCollection<Encoding> offspring;
            Selection selectionStrategy;
            Crossover crossoverStrategy;
        };

        template <typename FP>
        struct _DE_INF_WRAPPER {
            double (*original_evaluate)(std::vector<FP> &, void *);
            void *original_inf;
        };

        template <typename Encoding>
        struct Instance {
            uint64_t generationLimit;
            void *inf;
            // Neighbourhood generator: accepts an encoding and returns a vector of neighbourhood encodings.
            std::vector<Encoding> (*neighbourhood)(Encoding &);
            double (*evaluate)(Encoding &, void *);
            void (*mutate)(Encoding &, double);
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
        template <typename FP, typename... DEArgs>
        Solution<std::vector<FP>> evolution(Instance<std::vector<FP>> &, DE<DEArgs...> &,
                                            std::vector<std::vector<FP>> &);

        template <typename Encoding, typename Algorithm>
        Solution<Encoding> evolution(Instance<Encoding> &, Algorithm &, std::vector<Encoding> &);

        template <typename Encoding, typename... DEArgs>
        void initialise(Instance<Encoding> &, DE<DEArgs...> &, std::vector<Encoding> &);

        template <typename Encoding, typename... MAArgs>
        void initialise(Instance<Encoding> &, MA<Encoding, MAArgs...> &, std::vector<Encoding> &);

        template <typename Encoding>
        SolCollection<Encoding> initialisePopulation(Instance<Encoding> &, std::vector<Encoding> &);

        template <typename Encoding, typename... DEArgs>
        void generate(Instance<Encoding> &, SolCollection<Encoding> &, DE<DEArgs...> &);

        template <typename Encoding, typename... MAArgs>
        void generate(Instance<Encoding> &, SolCollection<Encoding> &, MA<Encoding, MAArgs...> &);

        template <typename Encoding, typename... MAArgs>
        inline void generate(Instance<Encoding> &, SolCollection<Encoding> &, MA<Encoding, MAArgs...> &);

        template <typename Encoding, typename... MAArgs>
        inline void mate(Instance<Encoding> &instance, SolCollection<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, MA<Encoding, MAArgs...> &);

        template <typename Encoding>
        inline size_t mateSelect(SolCollection<Encoding> &, Tournament &);

        template <typename Encoding>
        inline void crossover(Instance<Encoding> &instance, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, OP &);

        template <typename Encoding>
        inline void crossover(Instance<Encoding> &instance, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, OX &);

        template <typename Encoding>
        inline void crossover(Instance<Encoding> &instance, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, Solution<Encoding> &, double, PMX &);

        template <typename Encoding, typename... DEArgs>
        Encoding DE_mate(Encoding &, SolCollection<Encoding> &, DE<DEArgs...> &);

        template <typename Encoding, typename... DEArgs>
        Encoding DE_mateSelect(std::vector<Encoding> &, SolCollection<Encoding> &,
                                DE<DEArgs...> &, DE_Best &);

        template <typename Encoding, typename... DEArgs>
        Encoding DE_mateSelect(std::vector<Encoding> &, SolCollection<Encoding> &,
                                DE<DEArgs...> &, DE_Random &);

        template <typename Encoding, typename... DEArgs>
        Encoding DE_mateSelect(std::vector<Encoding> &, SolCollection<Encoding> &,
                                DE<DEArgs...> &, DE_CurrentToBest &);

        template <typename Encoding, typename... DEArgs>
        Encoding DE_mateSelect(std::vector<Encoding> &, SolCollection<Encoding> &,
                                DE<DEArgs...> &, DE_CurrentToRandom &);

        template <typename Encoding>
        Encoding DE_mutation(std::vector<Encoding> &, SolCollection<Encoding> &, double, uint8_t);

        template <typename Encoding>
        Encoding DE_crossover(Encoding &, Encoding &, double, DE_None &);

        template <typename Encoding>
        Encoding DE_crossover(Encoding &, Encoding &, double, DE_Binomial &);

        template <typename Encoding>
        Encoding DE_crossover(Encoding &, Encoding &, double, DE_Exponential &);

        template <typename FP>
        double _DE_EVALUATE_WRAPPER(std::valarray<FP> &, void *);
    }
}

// Definitions

template <typename Encoding>
inline MH::Solution<Encoding>::Solution() : encoding(), score(0) {}
template <typename Encoding>
inline MH::Solution<Encoding>::Solution(Encoding &e) : encoding(e), score(0) {}
template <typename Encoding>
inline MH::Solution<Encoding>::Solution(Encoding &e, double s) : encoding(e), score(s){}


template <typename Encoding, typename Selection, typename Crossover, typename LocalSearch, typename LSInstance>
MH::Evolutionary::MA<Encoding, Selection, Crossover, LocalSearch, LSInstance>::MA
    (size_t thePopulationSize, size_t theNumJobs, bool theElitism, bool theRemoveDuplicates,
     double theMutationProbability, LocalSearch &theLocalSearch, LSInstance &theLSInstance)
    : _offspringAreParents(false), elitism(theElitism), removeDuplicates(theRemoveDuplicates),
        mutationProbability(theMutationProbability), localSearch(theLocalSearch),
        lsInstance(theLSInstance) {
    offspring.resize(thePopulationSize);
    for(auto &elem : offspring) {
        elem.encoding.resize(theNumJobs);
    }
}

// The main search framework for trajectory-based algorithms
template <typename Encoding, typename AlgoType>
MH::Solution<Encoding>
MH::Trajectory::search(MH::Trajectory::Instance<Encoding> &instance,
                       AlgoType &algorithm,
                       Encoding &init) {

    MH::Trajectory::initialise(instance, algorithm, init);
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
        current = MH::Trajectory::select(instance, current, neighbours, algorithm);
        if(current < min) {
            min = current;
        }
    }
    return min;
}

// Initialise II.
template <typename Encoding, typename Strategy>
inline void
MH::Trajectory::initialise(MH::Trajectory::Instance<Encoding> &instance,
                           MH::Trajectory::II<Strategy> &ii,
                           Encoding &) {
    // Since II can update the generationLimit, it must be reset here.
    instance.generationLimit = ii.generationLimit;
}

// Initialise SA: set the temperature and epoch length.
template <typename Encoding>
inline void
MH::Trajectory::initialise(MH::Trajectory::Instance<Encoding> &,
                           MH::Trajectory::SA &sa,
                           Encoding &) {
    sa._temperature = sa.init_temperature;
    sa._epoch_count = 0;
}

// Initialise TS: fill the tabu list (queue).
template <typename Encoding, typename TraitType>
inline void
MH::Trajectory::initialise(MH::Trajectory::Instance<Encoding> &instance,
                           MH::Trajectory::TS<Encoding, TraitType> &ts,
                           Encoding & init) {
    ts._queue.resize(ts.length);
    std::fill(ts._queue.begin(), ts._queue.end(), ts.trait(init, instance.inf));
}

// Initialise RS: nothing to do here.
template <typename Encoding>
inline void
MH::Trajectory::initialise(MH::Trajectory::Instance<Encoding> &,
                           MH::Trajectory::RS &,
                           Encoding &) {
}

// II selection: call select_II based on II strategy.
template <typename Encoding, typename Strategy>
inline MH::Solution<Encoding> &
MH::Trajectory::select(MH::Trajectory::Instance<Encoding> &instance,
                       MH::Solution<Encoding> &current,
                       MH::SolCollection<Encoding> &neighbours,
                       MH::Trajectory::II<Strategy>& ii) {
    auto &result = MH::Trajectory::select_II(instance, current, neighbours, ii._strategy);
    ii.score = result.score;

    // Stop the search if at a local optimum.
    if(ii.score >= ii.prevScore) {
        instance.generationLimit = 0;
        ii.prevScore = std::numeric_limits<double>::infinity();
    }
    else {
        ii.prevScore = ii.score;
    }
    return result;
}

// SA selection: call select_SA and handle the cooling schedule
template <typename Encoding>
inline MH::Solution<Encoding> &
MH::Trajectory::select(MH::Trajectory::Instance<Encoding > &,
                       MH::Solution<Encoding> &current,
                       MH::SolCollection<Encoding> &neighbours,
                       MH::Trajectory::SA &sa) {
    auto &result = MH::Trajectory::select_SA(sa._temperature, current, neighbours);
    ++sa._epoch_count;
    //std::cout<<sa._temperature<<std::endl;
    if(sa._epoch_count == sa.epoch_length) {
        sa._temperature = sa.cooling(sa._temperature);
        sa._epoch_count = 0;
    }
    return result;
}

// TS selection: compare neighbours with the tabu list; choose the minimum not in the list
// and replace the oldest solution with the new solution.
template <typename Encoding, typename TraitType>
inline MH::Solution<Encoding> &
MH::Trajectory::select(MH::Trajectory::Instance<Encoding > &instance,
                       MH::Solution<Encoding> &,
                       MH::SolCollection<Encoding> &neighbours,
                       MH::Trajectory::TS<Encoding, TraitType> &ts) {
    auto &min = neighbours.front();
    for(auto &neighbour : neighbours) {
        if(std::find(ts._queue.begin(), ts._queue.end(),
                     ts.trait(neighbour.encoding, instance.inf)) == ts._queue.end() &&
           neighbour < min) {
            min = neighbour;
        }
    }
    ts._queue.pop_front();
    ts._queue.push_back(ts.trait(min.encoding, instance.inf));
    return min;
}

// Best improving II: select the minimum among the neighbours.
template <typename Encoding>
inline MH::Solution<Encoding> &
MH::Trajectory::select_II(MH::Trajectory::Instance<Encoding> &,
                          MH::Solution<Encoding> &current,
                          MH::SolCollection<Encoding> &neighbours,
                          II_BestImproving &) {
    auto &min = *std::min_element(neighbours.begin(), neighbours.end());
    return (min < current) ? min : current;
}

// First improving II: select the first solution that is better than current one during the iteration.
template <typename Encoding>
inline MH::Solution<Encoding> &
MH::Trajectory::select_II(MH::Trajectory::Instance<Encoding> &,
                          MH::Solution<Encoding> &current,
                          MH::SolCollection<Encoding> &neighbours,
                          II_FirstImproving &) {
    for(auto &neighbour : neighbours) {
        if(neighbour < current) {
            return neighbour;
        }
    }
    return current;
}

// Stochastic II: not yet implemented.
template <typename Encoding>
inline MH::Solution<Encoding> &
MH::Trajectory::select_II(MH::Trajectory::Instance<Encoding> &,
                          MH::Solution<Encoding> &current,
                          MH::SolCollection<Encoding> &,
                          II_Stochastic &) {
    return current;
}

// SA selection
template <typename Encoding>
inline MH::Solution<Encoding> &
MH::Trajectory::select_SA(double temperature,
                           MH::Solution<Encoding> &current,
                           MH::SolCollection<Encoding> &neighbours) {
    // random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> uniform;

    for(auto &neighbour : neighbours) {
        if((neighbour < current) ||
           (exp(current.score - neighbour.score) / temperature > uniform(eng)) ) {
            return neighbour;
        }
    }
    return current;
}

template <typename Encoding, typename Algorithm>
MH::Solution<Encoding>
MH::Evolutionary::evolution(MH::Evolutionary::Instance<Encoding> &instance,
                            Algorithm &algorithm,
                            std::vector<Encoding> &init) {
    MH::Evolutionary::initialise(instance, algorithm, init);
    auto population = MH::Evolutionary::initialisePopulation(instance, init);
    for(auto generationCount = 0UL;
        generationCount < instance.generationLimit;
        ++generationCount) {
        MH::Evolutionary::generate(instance, population, algorithm);
        std::cout<<"Generation "<<generationCount<<": ";
        auto minimum=population[0].score;
        for(auto &sol : population)
            if(sol.score<minimum){minimum=sol.score;}
        std::cout<<minimum<<std::endl;
    }
    auto min = *std::min_element(population.begin(), population.end());
    return min;
}

// Since DE will convert vectors to valarrays as an underlying type for performance,
// we need this wrapper to convert the initial population and restore the returned valarray.
template <typename FP, typename... DEArgs>
MH::Solution<std::vector<FP>>
MH::Evolutionary::evolution(Instance<std::vector<FP>> &instance,
                            DE<DEArgs...> &de,
                            std::vector<std::vector<FP>> &init) {
    // this wrapper wrap original evaluate function pointer and inf to new instance's inf
    _DE_INF_WRAPPER<FP> wrapper;
    wrapper.original_evaluate = instance.evaluate;
    wrapper.original_inf = instance.inf;

    // instance set to valarray type
    auto Uinstance = MH::Evolutionary::Instance<std::valarray<FP>>();
    // this evaluation function wrapper will restore original evaluator from inf
    Uinstance.evaluate = _DE_EVALUATE_WRAPPER;
    Uinstance.generationLimit = instance.generationLimit;
    Uinstance.inf = reinterpret_cast<void *>(&wrapper);

    // convert real vector to valarray
    std::vector<std::valarray<FP>> valarray_init(init.size());
    std::transform(init.begin(), init.end(),
                   valarray_init.begin(),
                   [&](auto &s) {
                       return std::valarray<FP>(s.data(), s.size());
                   });
    auto result = MH::Evolutionary::evolution(Uinstance, de, valarray_init);

    // convert vallarray back to real vector
    std::vector<FP> vec_result(std::begin(result.encoding), std::end(result.encoding));
    return MH::Solution<std::vector<FP>>(vec_result, result.score);
}

template <typename Encoding, typename... DEArgs>
inline void
MH::Evolutionary::initialise(MH::Evolutionary::Instance<Encoding> &,
                             MH::Evolutionary::DE<DEArgs...> &,
                             std::vector<Encoding> &) {
}

template <typename Encoding, typename... MAArgs>
inline void
MH::Evolutionary::initialise(MH::Evolutionary::Instance<Encoding> &,
                             MH::Evolutionary::MA<Encoding, MAArgs...> &,
                             std::vector<Encoding> &) {
}

template <typename Encoding>
inline MH::SolCollection<Encoding>
MH::Evolutionary::initialisePopulation(MH::Evolutionary::Instance<Encoding> &instance,
                                       std::vector<Encoding> &init) {
    MH::SolCollection<Encoding> population(init.size());
    std::transform(init.begin(), init.end(),
                   population.begin(),
                   [&](auto &s) {
                       return MH::Solution<Encoding>(s, instance.evaluate(s, instance.inf));
                   });
    return population;
}

template <typename Encoding, typename... MAArgs>
inline void
MH::Evolutionary::generate(Instance<Encoding> &instance,
                           MH::SolCollection<Encoding> &population,
                           MH::Evolutionary::MA<Encoding, MAArgs...> &ma) {
    auto &thePopulation = (ma._offspringAreParents) ? ma.offspring : population;
    auto &theOffspring = (ma._offspringAreParents) ? population : ma.offspring;
    for(size_t i = 0; i < population.size(); i += 2) {
        MH::Evolutionary::mate(instance, thePopulation, theOffspring[i], theOffspring[i + 1], ma);
        // local search
        theOffspring[i] = MH::Trajectory::search(ma.lsInstance, ma.localSearch, theOffspring[i].encoding);
        theOffspring[i + 1] = MH::Trajectory::search(ma.lsInstance, ma.localSearch, theOffspring[i + 1].encoding);
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
inline void
MH::Evolutionary::mate(Instance<Encoding> &instance,
                       MH::SolCollection<Encoding> &population,
                       MH::Solution<Encoding> &offspring1,
                       MH::Solution<Encoding> &offspring2,
                       MH::Evolutionary::MA<Encoding, MAArgs...> &ma) {
    auto parent1 = MH::Evolutionary::mateSelect(population, ma.selectionStrategy);
    auto parent2 = MH::Evolutionary::mateSelect(population, ma.selectionStrategy);
    while(parent2 == parent1) {
        parent2 = MH::Evolutionary::mateSelect(population, ma.selectionStrategy);
    }
    MH::Evolutionary::crossover(instance, population[parent1], population[parent2], offspring1, offspring2, ma.mutationProbability, ma.crossoverStrategy);
}

template <typename Encoding>
inline size_t
MH::Evolutionary::mateSelect(MH::SolCollection<Encoding> &population,
                             MH::Evolutionary::Tournament &tournament) {
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
inline void
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            MH::Solution<Encoding> &parent1,
                            MH::Solution<Encoding> &parent2,
                            MH::Solution<Encoding> &offspring1,
                            MH::Solution<Encoding> &offspring2,
                            double mutationProbability,
                            MH::Evolutionary::OP &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

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
}

// OX: encoding is limited to job indices.
template <typename Encoding>
inline void
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            MH::Solution<Encoding> &parent1,
                            MH::Solution<Encoding> &parent2,
                            MH::Solution<Encoding> &offspring1,
                            MH::Solution<Encoding> &offspring2,
                            double mutationProbability,
                            MH::Evolutionary::OX &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

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
inline void
MH::Evolutionary::crossover(Instance<Encoding> &instance,
                            MH::Solution<Encoding> &parent1,
                            MH::Solution<Encoding> &parent2,
                            MH::Solution<Encoding> &offspring1,
                            MH::Solution<Encoding> &offspring2,
                            double mutationProbability,
                            MH::Evolutionary::PMX &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());

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
}

template <typename Encoding, typename... DEArgs>
inline void
MH::Evolutionary::generate(Instance<Encoding> &instance,
                           std::vector<Solution<Encoding>> &population,
                           MH::Evolutionary::DE<DEArgs...> &de) {
    for(auto i = 0UL; i < population.size(); ++i) {
        auto target_vec = population[i].encoding;
        auto trial_vec = MH::Evolutionary::DE_mate(target_vec, population, de);
        // environment selection
        auto trial_score = instance.evaluate(trial_vec, instance.inf);
        if(trial_score < population[i].score) {
            population[i] = MH::Solution<Encoding>(trial_vec, trial_score);
        }
    }
}

template <typename Encoding, typename... DEArgs>
inline Encoding
MH::Evolutionary::DE_mate(Encoding &target_vec,
                          std::vector<Solution<Encoding>> &population,
                          MH::Evolutionary::DE<DEArgs...> &de) {
    std::vector<Encoding> selectionPool;
    selectionPool.push_back(target_vec);
    auto mutant_vec = MH::Evolutionary::DE_mateSelect(selectionPool, population, de, de._selection_strategy);
    mutant_vec += MH::Evolutionary::DE_mutation(selectionPool, population, de.scaling_factor, de.num_of_diff_vectors);
    auto trial_vec = MH::Evolutionary::DE_crossover(target_vec, mutant_vec, de.crossover_rate, de._crossover_strategy);
    return trial_vec;
}

template <typename Encoding, typename... DEArgs>
inline Encoding
MH::Evolutionary::DE_mateSelect(std::vector<Encoding> &selectionPool,
                                 MH::SolCollection<Encoding> &population,
                                 MH::Evolutionary::DE<DEArgs...> &,
                                 MH::Evolutionary::DE_Random &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<size_t> uniform(0, population.size() - 1);

    selectionPool.push_back(population[uniform(eng)].encoding);
    return selectionPool.back();
}

template <typename Encoding, typename... DEArgs>
inline Encoding
MH::Evolutionary::DE_mateSelect(std::vector<Encoding> &selectionPool,
                                 MH::SolCollection<Encoding> &population,
                                 MH::Evolutionary::DE<DEArgs...> &,
                                 MH::Evolutionary::DE_Best &) {
    selectionPool.push_back(std::min_element(population.begin(), population.end())->encoding);
    return selectionPool.back();
}

template <typename Encoding, typename... DEArgs>
inline Encoding
MH::Evolutionary::DE_mateSelect(std::vector<Encoding> &selectionPool,
                                 MH::SolCollection<Encoding> &population,
                                 MH::Evolutionary::DE<DEArgs...> &de,
                                 MH::Evolutionary::DE_CurrentToRandom &) {
    // random number generator
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<size_t> uniform(0, population.size() - 1);

    selectionPool.push_back(population[uniform(eng)].encoding);
    return selectionPool.front() + de.current_factor * (selectionPool.back() - selectionPool.front());
}

template <typename Encoding, typename... DEArgs>
inline Encoding
MH::Evolutionary::DE_mateSelect(std::vector<Encoding> &selectionPool,
                                 MH::SolCollection<Encoding> &population,
                                 MH::Evolutionary::DE<DEArgs...> &de,
                                 MH::Evolutionary::DE_CurrentToBest &) {
    selectionPool.push_back(std::min_element(population.begin(), population.end())->encoding);
    return selectionPool.front() + de.current_factor * (selectionPool.back() - selectionPool.front());
}

template <typename Encoding>
Encoding
MH::Evolutionary::DE_mutation(std::vector<Encoding> &selectionPool,
                         MH::SolCollection<Encoding> &population,
                         double scaling_factor,
                         uint8_t diff_vecs) {
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
    return scaling_factor * mutant_vec;
}

// No crossover; do nothing.
template <typename Encoding>
inline Encoding
MH::Evolutionary::DE_crossover(Encoding &,
                               Encoding &mutant_vec,
                               double,
                               MH::Evolutionary::DE_None &) {
    return mutant_vec;
}

// DE binomial crossover
template <typename Encoding>
Encoding
MH::Evolutionary::DE_crossover(Encoding &target_vec,
                               Encoding &mutant_vec,
                               double crossover_rate,
                               MH::Evolutionary::DE_Binomial &) {
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
Encoding
MH::Evolutionary::DE_crossover(Encoding &target_vec,
                               Encoding &mutant_vec,
                               double crossover_rate,
                               MH::Evolutionary::DE_Exponential &) {
    // Random number generators
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> uniform_r;
    static std::uniform_int_distribution<size_t> uniform_i(0, mutant_vec.size());

    // Perform the crossover based on the target vector.
    auto trial_vec(target_vec);

    // Generate a random crossover starting position.
    auto pos = uniform_i(eng);
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
