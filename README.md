# metaheuristic

##Trajectory Algorithms:

###呼叫
```
MH::Trajectory::search(instance, algortihm, init_solution);
```

###建立Instance：
```
auto instance = MH::Trajectory::Instance<Encoding>();
```
* `Encoding`是解編碼的類型。

###設定Instance：
* `instance.generationLimit` : 搜尋代數限制。
 * 原型`uint64_t generationLimit;`
* `instance.neighborhood` : 產生鄰域解的函式(neighbourhood generator)。
 * 原型`std::vector<Encoding> (*neighbourhood)(Encoding &);`
* `instance.evaluate` : 計算解分數的函式(evaluator)。
 * 原型`double (*evaluate)(Encoding &, void *inf);`
* `instance.inf` : 提供給evaluator的額外資訊
 * `void *inf;`
 * 請使用`reinterpret_cast<void *>()`轉型

###建立algorithm：
II :
```
auto II = MH::Trajectory::II<SelectionStrategy>();
```
或使用別名`MH::Trajectory::ItrativeImprovement`

* `SelectionStrategy`是II的選擇策略，提供兩種。
 * `MH::Trajectory::II_BestImproving`
 * `MH::Trajecotry::II_FirstImproving`

SA :
```
auto SA = MH::Trajectory::SA();
```
或使用別名`MH::Trajectory::SimulatedAnnealing`

TS :
```
auto SA = MH::Trajectory::TS<Encoding, TraitType>();
```
或使用別名`MH::Trajectory::TabuSearch`
* `Encoding`是解編碼的型別。
* `TraitType`是禁忌列表(tabu list)儲存的特徵型別。

###設定algorithm
II :
無

SA :
* `SA.epoch_length` : epoch長度。
 * `uint64_t epoch_length;` 
* `SA.init_temperature` : 初始溫度。
 * `double init_temperature;` 
* `SA.cooling` : 降溫函式。
 * `double (*cooling)(double);`

TS :
* `TS.length` : 禁忌列表長度。
 * `uint8_t length;`
* `TS.trait` : 特徵轉換函式。
 * `TraitType (*trait)(Encoding&, void *);`
 * 注意第二個參數為`instance.inf`

## Evolutioanry Algorithms

###呼叫
```
MH::Evolutionary::evolution(instance, algorithm, init_solution)
```

###建立Instance
```
auto instance = MH::Evolutionary::Instance<Encoding>();
```
* `Encoding`是解編碼的類型。

###設定Instance：
* `instance.generationLimit` : 搜尋代數限制。
 * 原型`uint64_t generationLimit;`
* `instance.neighborhood` : 產生鄰域解的函式(neighbourhood generator)。
 * 原型`std::vector<Encoding> (*neighbourhood)(Encoding &);`
* `instance.evaluate` : 計算解分數的函式(evaluator)。
 * 原型`double (*evaluate)(Encoding &, void *inf);`
* `instance.inf` : 提供給evaluator的額外資訊
 * `void *inf;`
 * 請使用`reinterpret_cast<void *>()`轉型

###建立algorithm：
DE :
```
auto DE = MH::Evolutionary::DE<Selection, Crossover>();
```
或使用別名`MH::Evolutionary::DifferentialEvolution`

* `Selection`是DE的選擇策略，提供四種。
 * `MH::Evolutionary::DE_Random`
 * `MH::Evolutionary::DE_Best`
 * `MH::Evolutionary::DE_CurrentToRandom`
 * `MH::Evolutionary::DE_CurrentToBest`
* `Crossover`是DE的交配策略，提供兩種。
 * `MH::Evolutionary::DE_Binomial`
 * `MH::Evolutionary::DE_Exponential`

###設定algorithm
DE :
* `DE.crossover_rate` : 交配機率，必須介於區間(0, 1]。
 * `double crossover_rate;` 
* `DE.current_factor` : ToCurrent策略的縮放倍率。
 * 只有使用`DE_CurrentToRandom`或`DE_CurrentToBest`才須設定。
 * `double current_factor;`
* `DE.scaling_factor` : mutant vector的伸縮倍率。
 * `double scaling_factor;`
* `DE.num_of_diff_vectors` : 產生的mutant vector數量。
 * `uint8_t num_of_diff_vectors;`

###其他
關於Encoding:
必須提供`operator==()`的重載版本。
