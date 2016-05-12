#include "metaheuristic.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

// aliases
namespace MH = MetaHeuristics;
typedef std::vector<uint8_t> Permutation;
typedef std::vector<std::vector<uint16_t>> Table;

std::vector<std::vector<uint16_t>> PFSPparse_data(std::fstream&);
std::vector<Permutation> PFSPneighbors(Permutation&);
double PFSPmakespan(Permutation&, void*);
Permutation PFSPconvert(Permutation &encoding, void *);

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage : ./PFSP [test_data]" << std::endl;
        exit(-1);
    }
    std::cout << "Opening " << argv[1] << "... ";

    std::fstream file;
    try {
        file.open(argv[1], std::ios::in);
    }
    catch (std::ifstream::failure &e) {
        std::cerr << "An error occur while opening file." << std::endl;
        std::cerr << "Please make sure the file name is avlid and isn't in use." << std::endl;
        exit(-1);
    }
    std::cout << "Success" << std::endl;

    std::cout << "Parsing data... ";
    auto time_table = PFSPparse_data(file);
    auto num_machines = time_table.size();
    auto num_tasks = time_table.front().size();
    std::cout << "done" << std::endl;
    std::cout << "Number of tasks : " << num_tasks << std::endl;
    std::cout << "Number of machines : " << num_machines << std::endl;
    
    // Configure problem instance for trajectory based metaheuristics
    auto Tinstance = MH::Trajectory::InstanceType<Permutation>();
    Tinstance.generation_limit = 3000;
    Tinstance.neighbors = PFSPneighbors;
    Tinstance.evaluate = PFSPmakespan;
    Tinstance.inf = reinterpret_cast<void *>(&time_table);
    
    // II_FirstImproving | II_BestImproving | II_Stochastic
    //auto II = MH::Trajectory::IterativeImprovement<MH::Trajectory::II_FirstImproving>();

    /*
    auto SA = MH::Trajectory::SimulatedAnnealing();
    SA.init_temperature = 10000;
    SA.epoch_length = 20;
    */

    auto TS = MH::Trajectory::TabuSearch<Permutation, Permutation>();
    TS.length = 70;
    TS.trait = PFSPconvert;


    Permutation init(num_tasks);
    std::iota(init.begin(), init.end(), 1);
    std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    //MH::Trajectory::search(Tinstance, TS, init);
    
    for(auto i = 0; i < 20; ++i) {
        MH::Trajectory::search(Tinstance, TS, init);
        std::shuffle(init.begin(), init.end(), eng);
    }
    /*
    for(auto i = 0; i < 20; ++i) {
        MH::Trajectory::search(Tinstance, SA, init);
        std::shuffle(init.begin(), init.end(), eng);
    }
    */
    return 0;
}

std::vector<std::vector<uint16_t>> PFSPparse_data(std::fstream &file) {
    uint16_t num_tasks, num_machines;
    file >> num_tasks >> num_machines;
    file.ignore(std::numeric_limits<int64_t>::max(), '\n');

    Table time_table(num_machines);
    for(auto &row : time_table) {
        row.resize(num_tasks);
        for(auto &entry : row) {
            file >> entry;
        }
    }
    return time_table;
}

std::vector<Permutation> PFSPneighbors(Permutation &perm) {
    static std::default_random_engine eng(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::vector<Permutation> neighbors(perm.size() - 1);
    uint16_t count = 0;
    for(auto &neighbor : neighbors) {
        neighbor.resize(perm.size());
        std::copy(perm.begin(), perm.end(), neighbor.begin());
        std::swap(neighbor[count], neighbor.back());
        ++count;
    }
    // std::random_shuffle() is deprecated
    std::shuffle(neighbors.begin(), neighbors.end(), eng);
    return neighbors;
}

double PFSPmakespan(Permutation &perm, void *inf) {
    Table time_table = *reinterpret_cast<Table *>(inf);
    auto num_machines = time_table.size();
    auto num_tasks = time_table.front().size();
    std::vector<uint32_t> makespan_table(num_tasks);

    for(size_t machine_idx = 0; machine_idx < num_machines; ++machine_idx) {
        makespan_table[0] += time_table[machine_idx][perm[0] - 1];
        for(size_t task_idx = 1; task_idx < num_tasks; ++task_idx) {
            makespan_table[task_idx] = makespan_table[task_idx] < makespan_table[task_idx - 1] ?
                makespan_table[task_idx - 1] + time_table[machine_idx][perm[task_idx] - 1] :
                makespan_table[task_idx] + time_table[machine_idx][perm[task_idx] - 1];
        }
    }
    return makespan_table.back();
}

inline double PFSPcooling(double temperature) {
    return temperature * 0.95;
}

inline Permutation PFSPconvert(Permutation &encoding, void *) {
    return encoding;
}
