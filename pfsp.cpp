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

std::vector<std::vector<uint16_t>> PFSPParseData(std::fstream&);
std::vector<Permutation> PFSPSwapNeighbourhood(Permutation&);
double PFSPMakespan(Permutation&, void*);
Permutation PFSPConvert(Permutation &encoding, void *);
double DETest(std::vector<double> &, void *);

int main(int argc, char** argv) {
    if(argc < 2) {
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
    TInstance.neighbourhood = PFSPSwapNeighbourhood;
    TInstance.evaluate = PFSPMakespan;
    TInstance.inf = reinterpret_cast<void *>(&timeTable);
    
    // II_FirstImproving | II_BestImproving | II_Stochastic
    auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>();
    //auto SA = MH::Trajectory::SimulatedAnnealing();
    //SA.init_temperature = 10000;
    //SA.epoch_length = 20;
    //auto TS = MH::Trajectory::TabuSearch<Permutation, Permutation>();
    //TS.length = 70;
    //TS.trait = PFSPConvert;

/*
    // Generate initial solution for trajectory-based metaheuristics.
    Permutation init(numJobs);
    std::iota(init.begin(), init.end(), 1);
*/

    // Configure the problem instance for evolutionary algorithms.
    auto Einstance = MH::Evolutionary::Instance<Permutation>();
    Einstance.generationLimit = 7000;
    Einstance.evaluate = PFSPMakespan;
    Einstance.inf = reinterpret_cast<void *>(&timeTable);

    // Configure a memetic algorithm.
    auto MA = MH::Evolutionary::MemeticAlgorithm<Permutation, MH::Evolutionary::Tournament, MH::Evolutionary::OP, MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>, MH::Trajectory::Instance<Permutation>>(100, numJobs, II, TInstance);
    MA.selectionStrategy.size = 2;

    // random engine
    std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    // Generate initial population.
    std::vector<Permutation> init(100);
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

    //MH::Trajectory::search(TInstance, TS, init);
    MH::Evolutionary::evolution(Einstance, MA, init);
    
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

std::vector<Permutation> PFSPSwapNeighbourhood(Permutation &perm) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::vector<Permutation> neighbours(perm.size() - 1);
    uint16_t count = 0;
    for(auto &neighbour : neighbours) {
        neighbour.resize(perm.size());
        std::copy(perm.begin(), perm.end(), neighbour.begin());
        std::swap(neighbour[count], neighbour.back());
        ++count;
    }
    // std::random_shuffle() is deprecated
    std::shuffle(neighbours.begin(), neighbours.end(), eng);
    return neighbours;
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
