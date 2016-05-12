#pragma once

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <deque>
#include <valarray>
#include <unordered_set>

// Declarations
namespace MetaHeuristics {

    // All algorithms will use SolutionType to store the solution encoding and evaluation result (score).
    // Template parameter EncodingType is the type of the encoding suppose to use.
    template <typename EncodingType>
    struct SolutionType {
        friend bool operator<(SolutionType<EncodingType> &a, SolutionType<EncodingType> &b) {
            return a.score < b.score;
        }
        // Solution encoding
        EncodingType encoding;
        // Evaluation result
        double score;
        // Three types of constructor
        SolutionType();
        SolutionType(EncodingType &);
        SolutionType(EncodingType &, double);
    };

    namespace Trajectory {

        // A trajectory instance object specifies generation limit, neighborhood functions and evaluation function
        // there is also an optional "inf" field to provide additional information to evaluation function
        template <typename EncodingType>
        struct InstanceType {
            uint64_t generation_limit;
            // user provide neighborhood generate function
            // aceept an encoding and return a vector of neighborhood encodings
            std::vector<EncodingType> (*neighbors)(EncodingType &);
            // optional information provide to evaluate
            void *inf;
            // evaluate function, accept an encoding and additional information from instance.inf as (void *)
            // return a real number which is suppose to be minimize.
            double (*evaluate)(EncodingType &, void *);
        };


        template <typename II_StrategyType>
        struct IterativeImprovement{
            II_StrategyType strategy;
        };

        // These class is used as temeplate parameter for II
        class II_BestImproving {};
        class II_FirstImproving {};
        class II_Stochastic {};

        struct SimulatedAnnealing{
            double init_temperature;
            int64_t epoch_length;
            double (*cooling)(double);
            double temperature;
            int64_t epoch_count;
        };

        template <typename EncodingType, typename TraitType>
        struct TabuSearch{
            int8_t length;
            // the trait function accept an encoding and transform it into traits to store in tabu list.
            TraitType (*trait)(EncodingType&, void *);
            std::deque<TraitType> queue;
        };

        // Random search is aim to compare with others
        struct RandomSearch{};

        // aliases
        template <typename StrategyType>
        using II = IterativeImprovement<StrategyType>;
        using SA = SimulatedAnnealing;
        template <typename EncodingType, typename TraitType>
        using TS = TabuSearch<EncodingType, TraitType>;
        using RS = RandomSearch;

        // Function declarations
        template <typename EncodingType, typename AlgorithmType>
        SolutionType<EncodingType> search(InstanceType<EncodingType> &,
                                          AlgorithmType &,
                                          EncodingType &);
        template <typename EncodingType, typename StrategyType>
        void initialize(InstanceType<EncodingType> &, II<StrategyType> &, EncodingType &);
        template <typename EncodingType>
        void initialize(InstanceType<EncodingType> &, SA &, EncodingType &);
        template <typename EncodingType, typename TraitType>
        void initialize(InstanceType<EncodingType> &, TS<EncodingType, TraitType> &, EncodingType &);
        template <typename EncodingType>
        void initialize(InstanceType<EncodingType> &, RS &, EncodingType &);
        template <typename EncodingType, typename StrategyType>
        SolutionType<EncodingType>& select(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           II<StrategyType> &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           SA &);
        template <typename EncodingType, typename TraitType>
        SolutionType<EncodingType>& select(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           TS<EncodingType, TraitType> &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           RS &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select_II(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           II_BestImproving &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select_II(InstanceType<EncodingType> &,
                                           SolutionType<EncodingType> &,
                                           std::vector<SolutionType<EncodingType>> &,
                                           II_FirstImproving &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select_II(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          II_Stochastic &);
        template <typename EncodingType>
        SolutionType<EncodingType>& select_SA(double,
                                              SolutionType<EncodingType> &,
                                              std::vector<SolutionType<EncodingType>> &);
    }

    namespace Evolutionary {

        class DE_Random {};
        class DE_Best {};
        class DE_CurrentToRandom {};
        class DE_CurrentToBest {};

        class DE_Binomial {};
        class DE_Exponential {};

        // EncodingType of DE is restrict to vector of real numbers (float | double | long double)
        template <typename SelectionStrategy, typename CrossoverStrategy>
        struct DifferentialEvolution {
            double crossover_rate;
            double current_factor;
            double scaling_factor;
            uint8_t num_of_diff_vectors;
            SelectionStrategy selection_strategy;
            CrossoverStrategy crossover_strategy;
        };

        template <typename EncodingType>
        struct InstanceType {
            uint64_t generation_limit;
            void *inf;
            double (*evaluate)(EncodingType &, void *);
        };

        // aliases
        template <typename SelectionStrategy, typename CrossoverStrategy>
        using DE = DifferentialEvolution<SelectionStrategy, CrossoverStrategy>;

        // Function definitions
        template <typename FPprecision, typename SelectionStrategy, typename CrossoverStrategy>
        SolutionType<std::vector<FPprecision>> evolution(InstanceType<std::vector<FPprecision>> &,
                                             DifferentialEvolution<SelectionStrategy, CrossoverStrategy> &,
                                             std::vector<std::vector<FPprecision>> &);
        template <typename EncodingType, typename AlgorithmType>
        SolutionType<EncodingType> evolution(InstanceType<EncodingType> &,
                                             AlgorithmType &,
                                             std::vector<EncodingType> &);
        template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
        void initialize(InstanceType<EncodingType> &,
                        DE<SelectionStrategy, CrossoverStrategy> &,
                        std::vector<EncodingType> &);
        template <typename EncodingType>
        std::vector<SolutionType<EncodingType>> initialize_popultion(InstanceType<EncodingType> &,
                                                                     std::vector<EncodingType> &);
        template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
        std::vector<SolutionType<EncodingType>> generate(InstanceType<EncodingType> &,
                                                         std::vector<SolutionType<EncodingType>> &,
                                                         DE<SelectionStrategy, CrossoverStrategy> &);

        template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy> 
        EncodingType DE_mate(EncodingType &,
                             std::vector<SolutionType<EncodingType>> &,
                             DE<SelectionStrategy, CrossoverStrategy> &);
        template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
        EncodingType DE_mate_select(std::vector<EncodingType> &select_pool,
                                    std::vector<SolutionType<EncodingType>> &population,
                                    DE<SelectionStrategy, CrossoverStrategy> &,
                                    DE_Random &);
    }
}

// Definitions

namespace MH = MetaHeuristics;

template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType() : encoding(), score(0){}
template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType(EncodingType &e) : encoding(e), score(0){}
template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType(EncodingType &e, double s) : encoding(e), score(s){}

// The main search framework for trajectory-based algorithms
template <typename EncodingType, typename AlgoType>
MH::SolutionType<EncodingType>
MH::Trajectory::search(MH::Trajectory::InstanceType<EncodingType> &instance,
                       AlgoType &algorithm,
                       EncodingType &init) {

    MH::Trajectory::initialize(instance, algorithm, init);
    auto current = SolutionType<EncodingType>(init, instance.evaluate(init, instance.inf));
    auto min = current;

    for(uint64_t generation_count = 0;
        generation_count < instance.generation_limit;
        ++generation_count) {

        auto neighbors_encoding = instance.neighbors(current.encoding);
        std::vector<SolutionType<EncodingType>> neighbors(neighbors_encoding.size());
        // evaluate each encoding, and store to vector of SolutionType
        std::transform(neighbors_encoding.begin(),
                       neighbors_encoding.end(),
                       neighbors.begin(),
                       [&](auto &e) {
                           return SolutionType<EncodingType>(e, instance.evaluate(e, instance.inf));
                       });

        // each algorithm is different on there selection
        current = MH::Trajectory::select(instance, current, neighbors, algorithm);
        //std::cout << "Generation " << generation_count << " : " << current.score << std::endl;
        if(current < min) {
            min = current;
        }
    }
    std::cout << "Final value = " << min.score << std::endl;
    return min;
}

// initialize II, nothing to do here
template <typename EncodingType, typename StrategyType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::II<StrategyType> &,
                           EncodingType &) {
    std::cout << "Starting Iterative Improvement ..." << std::endl;
}

// initialize SA, set the temperature and epoch
template <typename EncodingType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::SA &sa,
                           EncodingType &) {
    std::cout << "Starting Simulated Annealing ..." << std::endl;
    sa.temperature = sa.init_temperature;
    sa.epoch_count = 0;
}

// initialize TS, fill the tabu list (queue)
template <typename EncodingType, typename TraitType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &instance,
                           MH::Trajectory::TS<EncodingType, TraitType> &ts,
                           EncodingType & init) {
    std::cout << "Starting Tabu Search ..." << std::endl;
    ts.queue.resize(ts.length);
    std::fill(ts.queue.begin(), ts.queue.end(), ts.trait(init, instance.inf));
}

// initialize RS, nothing to do here.
template <typename EncodingType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::RS &,
                           EncodingType &) {
    std::cout << "Starting Random Search ..." << std::endl;
}

// II selection, call select_II based on II strategy
template <typename EncodingType, typename StrategyType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select(MH::Trajectory::InstanceType<EncodingType> &instance,
                       MH::SolutionType<EncodingType> &current,
                       std::vector<MH::SolutionType<EncodingType>> &neighbors,
                       MH::Trajectory::II<StrategyType>& ii) {
    return MH::Trajectory::select_II(instance, current, neighbors, ii.strategy);
}

// SA selection, call select_SA and handle the cooling schedule
template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select(MH::Trajectory::InstanceType<EncodingType > &,
                       MH::SolutionType<EncodingType> &current,
                       std::vector<MH::SolutionType<EncodingType>> &neighbors,
                       MH::Trajectory::SA &sa) {
    auto &result = MH::Trajectory::select_SA(sa.temperature, current, neighbors);
    ++sa.epoch_count;
    if(sa.epoch_count == sa.epoch_length) {
        sa.temperature = sa.cooling(sa.temperature);
        sa.epoch_count = 0;
    }
    return result;
}

// TS selection, compare neighbors with tabu list, choose the minimum not in the list
// and replace the oldest with the new solution
template <typename EncodingType, typename TraitType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select(MH::Trajectory::InstanceType<EncodingType > &instance,
                       MH::SolutionType<EncodingType> &,
                       std::vector<MH::SolutionType<EncodingType>> &neighbors,
                       MH::Trajectory::TS<EncodingType, TraitType> &ts) {
    auto &min = neighbors.front();
    for(auto &neighbor : neighbors) {
        if(std::find(ts.queue.begin(),
                     ts.queue.end(),
                     ts.trait(neighbor.encoding, instance.inf)) == ts.queue.end() &&
           neighbor < min) {
            min = neighbor;
        }
    }
    ts.queue.pop_front();
    ts.queue.push_back(ts.trait(min.encoding, instance.inf));
    return min;
}

// Best improving II, select the minimum among neighbors
template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select_II(MH::Trajectory::InstanceType<EncodingType> &,
                          MH::SolutionType<EncodingType> &current,
                          std::vector<MH::SolutionType<EncodingType>> &neighbors,
                          II_BestImproving &) {
    auto &min = *std::min(neighbors.begin(), neighbors.end()); 
    return (min < current) ? min : current;
}

// First improving II, select the first solution that is better than current one during the iteration
template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select_II(MH::Trajectory::InstanceType<EncodingType> &,
                          MH::SolutionType<EncodingType> &current,
                          std::vector<MH::SolutionType<EncodingType>> &neighbors,
                          II_FirstImproving &) {
    for(auto &neighbor : neighbors) {
        if(neighbor < current) {
            return neighbor;
        }
    }
    return current;
}

// Stochastic II, not implement yet
template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select_II(MH::Trajectory::InstanceType<EncodingType> &,
                          MH::SolutionType<EncodingType> &current,
                          std::vector<MH::SolutionType<EncodingType>> &,
                          II_Stochastic &) {
    return current;
}

// SA selection
template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select_SA(double temperature,
                           MH::SolutionType<EncodingType> &current,
                           std::vector<MH::SolutionType<EncodingType>> &neighbors) {
    // uniform random number generators
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    static std::uniform_real_distribution<double> uniform;
    for(auto &neighbor : neighbors) {
        if((neighbor < current) ||
           (exp(current.score - neighbor.score) / temperature > uniform(eng)) ) {
            return neighbor;
        }
    }
    return current;
}


template <typename EncodingType, typename AlgorithmType>
MH::SolutionType<EncodingType>
MH::Evolutionary::evolution(MH::Evolutionary::InstanceType<EncodingType> &instance,
                            AlgorithmType &algorithm,
                            std::vector<EncodingType> &init) {
    MH::Evolutionary::initialize(instance, algorithm, init);
    auto population = MH::Evolutionary::initialize_popultion(instance, init);
    for(auto generation_count = 0UL;
        generation_count < instance.generation_limit;
        ++generation_count) {
        population = MH::Evolutionary::generate(instance, population, algorithm);
    }
    auto min = *std::min(population.begin(), population.end());
    std::cout << "Final value = " << min.score << std::endl;
    return min;
}

// Since DE will convert vector to valarray as an underlying type for performance,
// we need this wrapper to convert the initial population and restore the returned valarray
template <typename FPprecision, typename SelectionStrategy, typename CrossoverStrategy>
MH::SolutionType<std::vector<FPprecision>>
MH::Evolutionary::evolution(InstanceType<std::vector<FPprecision>> &instance,
                            DifferentialEvolution<SelectionStrategy, CrossoverStrategy> &de,
                            std::vector<std::vector<FPprecision>> &init) {
    // A new instance set to valarray type
    auto Uinstance = MH::Evolutionary::InstanceType<std::valarray<FPprecision>>();
    Uinstance.evaluate = instance.evaluate;
    Uinstance.generation_limit = instance.generation_limit;
    Uinstance.inf = instance.inf;
    // convert real vector to valarray
    std::vector<std::valarray<FPprecision>> valarray_init(init.size());
    std::transform(init.begin(),
                   init.end(),
                   valarray_init.begin(),
                   [&](auto &s) {
                       return std::valarray<FPprecision>(s.data(), s.size());
                   });
    auto &result = MH::Evolutionary::evolution(Uinstance, de, valarray_init);
    // convert vallarray back to real vector
    std::vector<FPprecision> vec_result(result.encoding.begin(), result.encoding.end());
    return MH::SolutionType<std::vector<FPprecision>>(vec_result, result.score);
}

template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
inline void
MH::Evolutionary::initialize(MH::Evolutionary::InstanceType<EncodingType> &,
                             MH::Evolutionary::DE<SelectionStrategy, CrossoverStrategy> &,
                             std::vector<EncodingType> &) {
    std::cout << "Starting Differential Evolution ... " << std::endl;
}


template <typename EncodingType>
std::vector<MH::SolutionType<EncodingType>>
MH::Evolutionary::initialize_popultion(MH::Evolutionary::InstanceType<EncodingType> &instance,
                                       std::vector<EncodingType> &init) {
    std::vector<MH::SolutionType<EncodingType>> population;
    std::transform(init.begin(),
                   init.end(),
                   population.begin(),
                   [&](auto &s) {
                       return MH::SolutionType<EncodingType>(s, instance.evaluate(s, instance.addinf));
                   });
    return population;
}

template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
std::vector<MH::SolutionType<EncodingType>>
MH::Evolutionary::generate(InstanceType<EncodingType> &instance,
                           std::vector<SolutionType<EncodingType>> &population,
                           MH::Evolutionary::DE<SelectionStrategy, CrossoverStrategy> &de) {
    for(auto i = 0UL; i < population.szie(); ++i) {
        auto &target_vec = population[i].encoding;
        auto &trial_vec = MH::Evolutionary::DE_mate(instance, target_vec, population, de);
        population[i] = MH::Evolutionary::env_select(instance, target_vec, trial_vec);
    }
}


template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy> 
EncodingType
MH::Evolutionary::DE_mate(EncodingType &target_vec,
                          std::vector<SolutionType<EncodingType>> &population,
                          MH::Evolutionary::DE<SelectionStrategy, CrossoverStrategy> &de) {
    std::vector<EncodingType> select_pool;
    select_pool.push_back(target_vec);
    auto &mutant_vec = MH::Evolutioanry::DE_mate_select(select_pool, population, de, de.selection_strategy);
    mutant_vec += MH::Evolutionary::DE_mutation(select_pool, population, de.scaling_factor, de.num_of_diff_vectors);
    auto &trial_vec = MH::Evolutionary::DE_crossover(target_vec, mutant_vec, de.crossover_rate, de.crossover_strategy);
    return target_vec;
}

template <typename EncodingType, typename SelectionStrategy, typename CrossoverStrategy>
EncodingType
MH::Evolutionary::DE_mate_select(std::vector<EncodingType> &select_pool,
                                 std::vector<MH::SolutionType<EncodingType>> &population,
                                 MH::Evolutionary::DE<SelectionStrategy, CrossoverStrategy> &,
                                 MH::Evolutionary::DE_Random &) {

}
