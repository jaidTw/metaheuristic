# metaheuristic

This is a simple metaheuristic library design for class

## Trajectory Algorithms:

### Crease Instance：
```
auto instance = MH::Trajectory::Instance<Encoding>();
```
* `Encoding` is the type of the encoded solution.

### Configure Instance：
* `instance.generationLimit` : Limit of search generations.
 * prototype `uint64_t generationLimit;`
* `instance.neighborhood` : neighbourhood generator.
 * prototype `std::vector<Encoding> (*neighbourhood)(Encoding &);`
* `instance.evaluate` : solution fitness evaluator.
 * prototype `double (*evaluate)(Encoding &, void *inf);`
* `instance.inf` : additional information provide to evaluator.
 * prototype `void *inf;`

### Create algorithm：
ItrativeImprovement (II) :
```
auto II = MH::Trajectory::II<SelectionStrategy>();
```
or use the alias`MH::Trajectory::ItrativeImprovement`

* `SelectionStrategy`is the seleciton strategy of II.
 * `MH::Trajectory::II_BestImproving`
 * `MH::Trajecotry::II_FirstImproving`

Simulated Annealing (SA) :
```
auto SA = MH::Trajectory::SA();
```
or use the alias`MH::Trajectory::SimulatedAnnealing`

TabuSearch (TS) :
```
auto TS = MH::Trajectory::TS<Encoding, TraitType>();
```
or use the alias`MH::Trajectory::TabuSearch`
* `Encoding` is the type of the encoded solution.
* `TraitType`is the type of the trait that will be stored in the tabu list.

### Configure Algorithm
II :
None

SA :
* `SA.epoch_length` : epoch length.
 * `uint64_t epoch_length;` 
* `SA.init_temperature` : initial temperature.
 * `double init_temperature;` 
* `SA.cooling` : cooling function.
 * `double (*cooling)(double);`

TS :
* `TS.length` : tabu tenure.
 * `uint8_t length;`
* `TS.trait` : trait convert function.
 * `TraitType (*trait)(Encoding&, void *);`
 * The second parameter is `instance.inf`

### Call
```
instance.search(algortihm, init_solution);
```

## Evolutioanry Algorithms

### Create Instance
```
auto instance = MH::Evolutionary::Instance<Encoding>();
```
* `Encoding` is the type of the encoded solution.

### Configure Instance：
* `instance.generationLimit` : limit of search generation。
 * prototype `uint64_t generationLimit;`
* `instance.neighborhood` : neighbourhood generator.
 * prototype `std::vector<Encoding> (*neighbourhood)(Encoding &);`
* `instance.evaluate` : solution fitness evaluator.
 * prototype `double (*evaluate)(Encoding &, void *inf);`
* `instance.inf` : additional information provide to evaluator.
 * `void *inf;`

### Create algorithm：
Differential Evolution(DE) :
```
auto DE = MH::Evolutionary::DE<Selection, Crossover>();
```
or use the alias `MH::Evolutionary::DifferentialEvolution`

* `Selection`is the selection strategy of DE.
 * `MH::Evolutionary::DE_Random`
 * `MH::Evolutionary::DE_Best`
 * `MH::Evolutionary::DE_CurrentToRandom`
 * `MH::Evolutionary::DE_CurrentToBest`
* `Crossover`is the crossover strategy of DE.
 * `MH::Evolutionary::DE_Binomial`
 * `MH::Evolutionary::DE_Exponential`

### Configure algorithm
DE :
* `DE.crossover_rate` : crossover rate, should be in the interval (0, 1].
 * `double crossover_rate;` 
* `DE.current_factor` : scaling factor of xToCurrent policy.
 * Only required for `DE_CurrentToRandom` or `DE_CurrentToBest`.
 * `double current_factor;`
* `DE.scaling_factor` : scaling factor of mutant vector.
 * `double scaling_factor;`
* `DE.num_of_diff_vectors` : the number of mutant vectors to be generated。
 * `uint8_t num_of_diff_vectors;`
 * 
### Call
```
instance.evolution(algorithm, init_solution)
```

## Others
About Encoding:
should provide the `operator==()`.
