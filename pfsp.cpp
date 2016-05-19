#include "metaheuristic.h"
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

// aliases
typedef std::vector<uint8_t> Permutation;
typedef std::vector<std::vector<uint16_t>> Table;
typedef std::chrono::high_resolution_clock Clock;

std::vector<std::vector<uint16_t>> PFSPParseData(std::fstream&);
std::vector<Permutation> PFSPInsertionNeighbourhoodSmall(Permutation&); // Supposedly preferable to swap.
std::vector<Permutation> PFSPInsertionNeighbourhood(Permutation&); // Slow.
std::vector<Permutation> PFSPSwapNeighbourhoodSmall(Permutation&);
void PFSPShiftMutationPerSolution(Permutation&, double);
void PFSPShiftMutationPerJob(Permutation&, double); // Terrible. Do not use.
double PFSPMakespan(Permutation&, void*); // Naïve algorithm. A faster version should be written for evaluating neighbourhoods.
Permutation PFSPConvert(Permutation &encoding, void *);
double DETest(std::vector<double> &, void *);

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: ./pfsp [test_data]" << std::endl;
        exit(-1);
    }
    std::cout << "Opening " << argv[1] << "... ";

    std::fstream file;
    try {
        file.open(argv[1], std::ios::in);
    }
    catch (std::ifstream::failure &e) {
        std::cerr << "An error occur while opening file." << std::endl;
        std::cerr << "Please make sure the file name is valid and isn't in use." << std::endl;
        exit(-1);
    }
    std::cout << "Success." << std::endl;

    std::cout << "Parsing data... ";
    auto timeTable = PFSPParseData(file);
    auto numMachines = timeTable.size();
    auto numJobs = timeTable.front().size();
    std::cout << "Done." << std::endl;
    std::cout << "Number of jobs: " << numJobs << std::endl;
    std::cout << "Number of machines: " << numMachines << std::endl;
    
    // Configure problem instance for trajectory-based metaheuristics.
    auto TInstance = MH::Trajectory::Instance<Permutation>();
    TInstance.generationLimit = 3000;
    TInstance.neighbourhood = PFSPInsertionNeighbourhoodSmall;
    TInstance.evaluate = PFSPMakespan;
    TInstance.inf = reinterpret_cast<void *>(&timeTable);
    
    // II_FirstImproving | II_BestImproving | II_Stochastic
    auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>(TInstance.generationLimit);
    auto SA = MH::Trajectory::SimulatedAnnealing();
    SA.init_temperature = 10000;
    SA.epoch_length = 20;
    auto TS = MH::Trajectory::TabuSearch<Permutation, Permutation>();
    TS.length = 70;
    TS.trait = PFSPConvert;

    // Configure the problem instance for evolutionary algorithms.
    auto EInstance = MH::Evolutionary::Instance<Permutation>();
    EInstance.generationLimit = 700;
    EInstance.evaluate = PFSPMakespan;
    EInstance.mutate = PFSPShiftMutationPerSolution;
    EInstance.inf = reinterpret_cast<void *>(&timeTable);

    // Configure a memetic algorithm.
    auto MA = MH::Evolutionary::MemeticAlgorithm<Permutation,
                MH::Evolutionary::Tournament,
                MH::Evolutionary::OP,
                MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>,
                    MH::Trajectory::Instance<Permutation>>
                (100, numJobs, true, true, 0.6, II, TInstance);
    // tournament size
    MA.selectionStrategy.size = 5;

    // random engine
    std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

/*
    // Generate initial solution for trajectory-based metaheuristics.
    Permutation init(numJobs);
    std::iota(init.begin(), init.end(), 1);
*/

    // Generate initial population.
    std::vector<Permutation> init(MA.offspring.size());
    for(auto &sol : init) {
        sol.resize(numJobs);
        std::iota(sol.begin(), sol.end(), 1);
        std::shuffle(sol.begin(), sol.end(), eng);
    }
    
/*
    auto DE = MH::Evolutionary::DifferentialEvolution<MH::Evolutionary::DE_Best, MH::Evolutionary::DE_Binomial>();
    DE.crossover_rate = 0.6;
    DE.scaling_factor = 0.6;
    DE.num_of_diff_vectors = 2;
    DE.current_factor = 0.5;
    // random engine use to shuffle init solution
    std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    std::vector<std::vector<double>> init(20);
    for(auto &sol : init) {
        sol.resize(30);
        std::iota(sol.begin(), sol.end(), 1);
        std::shuffle(sol.begin(), sol.end(), eng);
    }
*/

    auto start = Clock::now();
    //auto result = MH::Trajectory::search(TInstance, TS, init);
    auto result = MH::Evolutionary::evolution(EInstance, MA, init);
    std::cout << "\nFinal score: " << result.score << ".\n";
    std::cout << "花費的時間：";
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << duration.count() / 1000.0 << "秒。\n";
    
    return 0;
}

std::vector<std::vector<uint16_t>> PFSPParseData(std::fstream &file) {
    uint16_t numJobs, numMachines;
    file >> numJobs >> numMachines;
    file.ignore(std::numeric_limits<int64_t>::max(), '\n');

    Table timeTable(numMachines);
    for(auto &row : timeTable) {
        row.resize(numJobs);
        for(auto &entry : row) {
            file >> entry;
        }
    }
    return timeTable;
}

std::vector<Permutation> PFSPSwapNeighbourhoodSmall(Permutation &perm) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::vector<Permutation> neighbours(perm.size() - 1);
    uint16_t count = 0;
    for(auto &neighbour : neighbours) {
        neighbour.resize(perm.size());
        std::copy(perm.begin(), perm.end(), neighbour.begin());
        std::swap(neighbour[count], neighbour.back());
        ++count;
    }
    std::shuffle(neighbours.begin(), neighbours.end(), eng);
    return neighbours;
}

std::vector<Permutation> PFSPInsertionNeighbourhoodSmall(Permutation &perm) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::vector<Permutation> neighbours(perm.size() - 1);
    uint16_t count = 0;
    for(auto &neighbour : neighbours) {
        neighbour.resize(perm.size());
        std::copy(perm.begin(), perm.end(), neighbour.begin());
        neighbour.insert(neighbour.begin() + count, neighbour.back());
        neighbour.erase(neighbour.end() - 1);
        ++count;
    }
    std::shuffle(neighbours.begin(), neighbours.end(), eng);
    return neighbours;
}

std::vector<Permutation> PFSPInsertionNeighbourhood(Permutation &perm) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::vector<Permutation> neighbours((perm.size() - 1) * perm.size());
    size_t index = 0;
    for(size_t i = 0; i < perm.size(); ++i) {
        for(size_t j = i + 1; j < perm.size(); ++j) {
            neighbours[index].resize(perm.size());
            std::copy(perm.begin(), perm.end(), neighbours[index].begin());
            neighbours[index].insert(neighbours[index].begin() + j, neighbours[index][i]);
            neighbours[index].erase(neighbours[index].begin() + i); 
            ++index;
        }
    }
    for(size_t i = 0; i < perm.size(); ++i) {
        for(size_t j = i + 1; j < perm.size(); ++j) {
            neighbours[index].resize(perm.size());
            std::copy(perm.begin(), perm.end(), neighbours[index].begin());
            neighbours[index].insert(neighbours[index].end() - j, neighbours[index][perm.size() - i - 1]);
            neighbours[index].erase(neighbours[index].end() - i - 1); 
            ++index;
        }
    }
    std::shuffle(neighbours.begin(), neighbours.end(), eng);
    return neighbours;
}

void PFSPShiftMutationPerSolution(Permutation &perm, double mutationProbability) {
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    double random;
    random = (double)eng() / (double)eng.max();
    if(random < mutationProbability) {
        size_t oldPos = eng() % perm.size();
        size_t newPos;
        do {
            newPos = eng() % perm.size();
        } while(newPos == oldPos);
        perm.insert(perm.begin() + newPos, perm[oldPos]);
        if(newPos > oldPos) {
            perm.erase(perm.begin() + oldPos);
        }
        else {
            perm.erase(perm.begin() + oldPos + 1);
        }
    }
}

void PFSPShiftMutationPerJob(Permutation &perm, double mutationProbability) {
    static std::minstd_rand eng(std::chrono::system_clock::now().time_since_epoch().count());
    double random;
    for(size_t i = 0; i < perm.size(); ++i) {
        random = (double)eng() / (double)eng.max();
        if(random < mutationProbability) {
            size_t newPos = eng() % perm.size();
            perm.insert(perm.begin() + newPos, perm[i]);
            if(newPos > i) {
                perm.erase(perm.begin() + i);
            }
            else {
                perm.erase(perm.begin() + i + 1);
            }
        }
    }
}

double PFSPMakespan(Permutation &perm, void *inf) {
    Table timeTable = *reinterpret_cast<Table *>(inf);
    auto numMachines = timeTable.size();
    auto numJobs = timeTable.front().size();
    std::vector<uint32_t> makespanTable(numJobs);

    for(size_t machineIdx = 0; machineIdx < numMachines; ++machineIdx) {
        makespanTable[0] += timeTable[machineIdx][perm[0] - 1];
        for(size_t taskIdx = 1; taskIdx < numJobs; ++taskIdx) {
            makespanTable[taskIdx] = makespanTable[taskIdx] < makespanTable[taskIdx - 1] ?
                makespanTable[taskIdx - 1] + timeTable[machineIdx][perm[taskIdx] - 1] :
                makespanTable[taskIdx] + timeTable[machineIdx][perm[taskIdx] - 1];
        }
    }
    return makespanTable.back();
}

inline double PFSPCooling(double temperature) {
    return temperature * 0.95;
}

inline Permutation PFSPConvert(Permutation &encoding, void *) {
    return encoding;
}

double DETest(std::vector<double>& sol, void *) {
    std::vector<double> csol(sol);
    std::for_each(csol.begin(), csol.end(), [&](auto &n) { n *= n; });
    return std::accumulate(csol.begin(), csol.end(), .0);
}
