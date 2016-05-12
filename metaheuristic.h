#pragma once

#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <deque>
#include <unordered_set>

namespace MetaHeuristics {

    // All algorithms will use SolutionType to store the solution encoding and evaluation result.
    // Template parameter EncodingType is the type of the encoding suppose to use.
    template <typename EncodingType>
    struct SolutionType {
        friend bool operator<(SolutionType<EncodingType> &a, SolutionType<EncodingType> &b) {
            return a.score < b.score;
        }
        EncodingType encoding;
        double score;
        SolutionType();
        SolutionType(EncodingType &);
        SolutionType(EncodingType &, double);
    };

    namespace Trajectory {

        class II_BestImproving {};
        class II_FirstImproving {};
        class II_Stochastic {};

        template <typename II_StrategyType>
        struct IterativeImprovement{
            II_StrategyType strategy;
        };
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
            TraitType (*trait)(EncodingType&, void *);
            std::deque<TraitType> queue;
        };
        class RandomSearch{};

        template <typename EncodingType>
        struct InstanceType {
            uint64_t generation_limit;
            std::vector<EncodingType> (*neighbors)(EncodingType &);
            void *inf;
            double (*evaluate)(EncodingType &, void *);
        };
        template <typename StrategyType>
        using II = IterativeImprovement<StrategyType>;
        using SA = SimulatedAnnealing;
        template <typename EncodingType, typename TraitType>
        using TS = TabuSearch<EncodingType, TraitType>;
        using RS = RandomSearch;

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
        SolutionType<EncodingType> select(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          II<StrategyType> &);
        template <typename EncodingType>
        SolutionType<EncodingType> select(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          SA &);
        template <typename EncodingType, typename TraitType>
        SolutionType<EncodingType> select(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          TS<EncodingType, TraitType> &);
        template <typename EncodingType>
        SolutionType<EncodingType> select(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          RS &);
        template <typename EncodingType>
        SolutionType<EncodingType> select_II(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          II_BestImproving &);
        template <typename EncodingType>
        SolutionType<EncodingType> select_II(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          II_FirstImproving &);
        template <typename EncodingType>
        SolutionType<EncodingType> select_II(InstanceType<EncodingType> &,
                                          SolutionType<EncodingType> &,
                                          std::vector<SolutionType<EncodingType>> &,
                                          II_Stochastic &);
        template <typename EncodingType>
        SolutionType<EncodingType> &select_SA(double,
                                              SolutionType<EncodingType> &,
                                              std::vector<SolutionType<EncodingType>> &);
    }

    namespace Evolutionary {}
}

namespace MH = MetaHeuristics;

template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType() : encoding(), score(0){}
template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType(EncodingType &e) : encoding(e), score(0){}
template <typename EncodingType>
inline MH::SolutionType<EncodingType>::SolutionType(EncodingType &e, double s) : encoding(e), score(s){}

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
        std::transform(neighbors_encoding.begin(),
                       neighbors_encoding.end(),
                       neighbors.begin(),
                       [&](auto &e) {
                           return SolutionType<EncodingType>(e, instance.evaluate(e, instance.inf));
                       });

        current = MH::Trajectory::select(instance, current, neighbors, algorithm);
        //std::cout << "Generation " << generation_count << " : " << current.score << std::endl;
        if(current < min) {
            min = current;
        }
    }

    std::cout << "Final value = " << min.score << std::endl;
    return min;
}

template <typename EncodingType, typename StrategyType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::II<StrategyType> &,
                           EncodingType &) {
    std::cout << "Starting Iterative Improvement ..." << std::endl;
}

template <typename EncodingType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::SA &sa,
                           EncodingType &) {
    std::cout << "Starting Simulated Annealing ..." << std::endl;
    sa.temperature = sa.init_temperature;
    sa.epoch_count = 0;
}

template <typename EncodingType, typename TraitType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &instance,
                           MH::Trajectory::TS<EncodingType, TraitType> &ts,
                           EncodingType & init) {
    std::cout << "Starting Tabu Search ..." << std::endl;
    ts.queue.resize(ts.length);
    std::fill(ts.queue.begin(), ts.queue.end(), ts.trait(init, instance.inf));
}

template <typename EncodingType>
inline void
MH::Trajectory::initialize(MH::Trajectory::InstanceType<EncodingType> &,
                           MH::Trajectory::RS &,
                           EncodingType &) {
    std::cout << "Starting Random Search ..." << std::endl;
}

template <typename EncodingType, typename StrategyType>
inline MH::SolutionType<EncodingType>
MH::Trajectory::select(MH::Trajectory::InstanceType<EncodingType> &instance,
                       MH::SolutionType<EncodingType> &current,
                       std::vector<MH::SolutionType<EncodingType>> &neighbors,
                       MH::Trajectory::II<StrategyType>& ii) {
    return MH::Trajectory::select_II(instance, current, neighbors, ii.strategy);
}

template <typename EncodingType>
inline MH::SolutionType<EncodingType>
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

template <typename EncodingType, typename TraitType>
inline MH::SolutionType<EncodingType>
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

template <typename EncodingType>
inline MH::SolutionType<EncodingType>
MH::Trajectory::select_II(MH::Trajectory::InstanceType<EncodingType> &,
                          MH::SolutionType<EncodingType> &current,
                          std::vector<MH::SolutionType<EncodingType>> &neighbors,
                          II_BestImproving &) {
    auto &min = *std::min(neighbors.begin(), neighbors.end()); 
    return (min < current) ? min : current;
}

template <typename EncodingType>
inline MH::SolutionType<EncodingType>
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

template <typename EncodingType>
inline MH::SolutionType<EncodingType>
MH::Trajectory::select_II(MH::Trajectory::InstanceType<EncodingType> &,
                          MH::SolutionType<EncodingType> &current,
                          std::vector<MH::SolutionType<EncodingType>> &,
                          II_Stochastic &) {
    return current;
}

template <typename EncodingType>
inline MH::SolutionType<EncodingType> &
MH::Trajectory::select_SA(double temperature,
                           MH::SolutionType<EncodingType> &current,
                           std::vector<MH::SolutionType<EncodingType>> &neighbors) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    static std::uniform_real_distribution<double> uniform;
    for(auto &neighbor : neighbors) {
        if( (neighbor < current) ||
            (exp(current.score - neighbor.score) / temperature > uniform(eng)) ) {
            return neighbor;
        }
    }
    return current;
}
