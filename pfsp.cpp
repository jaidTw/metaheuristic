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

inline double PFSPCooling(double temperature);

// aliases
typedef std::vector<uint8_t> Permutation;
typedef std::vector<std::vector<uint16_t>> Table;
typedef std::chrono::high_resolution_clock Clock;

Table PFSPParseData(std::fstream&);
std::vector<Permutation> PFSPInsertionNeighbourhoodSmall(Permutation&); // Supposedly preferable to swap.
std::vector<Permutation> PFSPInsertionNeighbourhood(Permutation&); // Slow.
std::vector<Permutation> PFSPSwapNeighbourhoodSmall(Permutation&);
void PFSPShiftMutationPerSolution(Permutation&, double);
void PFSPShiftMutationPerJob(Permutation&, double); // Terrible. Do not use.
double PFSPMakespan(Permutation&, void*); // Naïve algorithm. A faster version should be written for evaluating neighbourhoods.
Permutation PFSPConvert(Permutation &encoding, void *);

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
    TInstance.generationLimit = 300;
    TInstance.neighbourhood = PFSPInsertionNeighbourhoodSmall;
    TInstance.evaluate = PFSPMakespan;
    TInstance.inf = reinterpret_cast<void *>(&timeTable);

    // II_FirstImproving | II_BestImproving | II_Stochastic
#ifdef USE_II_FI
    auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>(TInstance.generationLimit);
#elif USE_II_BI
    auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_BestImproving>(TInstance.generationLimit);
#elif USE_II_SC
    auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_Stochastic>(TInstance.generationLimit);
#elif USE_SA
    auto SA = MH::Trajectory::SimulatedAnnealing();
    SA.init_temperature = 10000;
    SA.cooling = PFSPCooling;
    SA.epoch_length = 20;
#elif USE_TS
    auto TS = MH::Trajectory::TabuSearch<Permutation, Permutation>();
    TS.length = 70;
    TS.trait = PFSPConvert;
#endif // USE_II

    // Configure the problem instance for evolutionary algorithms.
    auto EInstance = MH::Evolutionary::Instance<Permutation>();
    EInstance.generationLimit = 700;
    EInstance.evaluate = PFSPMakespan;
    EInstance.mutate = PFSPShiftMutationPerSolution;
    EInstance.inf = reinterpret_cast<void *>(&timeTable);

    // Configure a memetic algorithm.
    auto MA = MH::Evolutionary::MemeticAlgorithm<Permutation, MH::Evolutionary::Tournament,
#ifdef USE_OP
        MH::Evolutionary::OP,
#elif UES_SJOX
        MH::Evolutionary::SJOX,
#endif // USE_OP
#ifdef USE_II_FI
        MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>,
#elif USE_II_BI
        MH::Trajectory::IterativeImprovement<MH::Trajectory::II_BestImproving>,
#elif USE_II_SC
        MH::Trajectory::IterativeImprovement<MH::Trajectory::II_Stochastic>,
#elif USE_SA
        MH::Trajectory::SimulatedAnnealing,
#elif USE_TS
        MH::Trajectory::TabuSearch<Permutation,Permutation>,
#endif // USE_II_FI
        MH::Trajectory::Instance<Permutation>>(100, numJobs,true,true,0.6,
#if defined(USE_II_FI) || defined(USE_II_BI) || defined(USE_II_SC)
            II,
#elif USE_SA
            SA,
#elif USE_TS
            TS,
#endif // USE_SA
            TInstance);
    MA.selectionStrategy.size = 2;

    // random engine
    std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    // Generate initial population.
    auto initInstance = MH::Trajectory::Instance<Permutation>();
    initInstance.generationLimit = 300;
    initInstance.neighbourhood = PFSPSwapNeighbourhoodSmall;
    initInstance.evaluate = PFSPMakespan;
    initInstance.inf = reinterpret_cast<void *>(&timeTable);

    auto initSA = MH::Trajectory::SA();
    initSA.epoch_length = 20;
    initSA.init_temperature = 7000;
    initSA.cooling = PFSPCooling;

    auto start = Clock::now();

    std::vector<Permutation> init(MA.offspring.size());
    for(auto &sol : init) {
        sol.resize(numJobs);
        std::iota(sol.begin(), sol.end(), 1);
        std::shuffle(sol.begin(), sol.end(), eng);

        sol = MH::Trajectory::search(initInstance, initSA, sol).encoding;
    }

    auto result = MH::Evolutionary::evolution(EInstance, MA, init);

    std::cout << "\nFinal score: " << result.score << ".\n";
    std::cout << "Soent：";
    auto end = Clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << duration.count() / 1000.0 << "秒。\n";
    for (auto &e : result.encoding)
        std::cout << (int)e << " ";
    std::cout <<std::endl;

    return 0;
}

Table PFSPParseData(std::fstream &file) {
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
