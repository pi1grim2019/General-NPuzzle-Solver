# General-NPuzzle-Solver

## Introduction
This project implements a general solver for the classic N-Puzzle problem (e.g., 8-puzzle or 15-puzzle) using three search algorithms: Uniform Cost Search (UCS), A* Search with the Misplaced Tile heuristic, and A* Search with the Manhattan Distance heuristic.

The solver is written in Python 3 and supports customizable puzzle dimensions (N × N). The program automatically generates or accepts user-defined puzzle configurations, performs state-space search, and outputs the optimal solution path along with experimental statistics.

## Running the code

You can run the code with: 
```
python main.py
```

### Program Interaction Example

```
Welcome to the N-Puzzle Solver!
Enter the puzzle dimension N (e.g., 3 for 8-puzzle, 4 for 15-puzzle): 3
Type '1' to use a randomly generated 8-puzzle, or '2' to create your own: 2
Enter the puzzle numbers (0 for blank), 3 numbers per row, row by row:
Enter row 1: 1 6 7
Enter row 2: 5__ 3
Enter row 3: 4 8 2
```

### Output Example

```

Running algorithms...

--- A* with Manhattan Distance ---
Solution found!
Depth-nodes data saved to output\AStar_with_Manhattan_Distance.depth_nodes.txt

Solution Path (A* with Manhattan Distance):
Step 0:
1 6 7
5__ 3
4 8 2
...
Step 16:
1 2 3
4 5 6
7 8 __

--- Algorithm Comparison ---
Algorithm                 Depth   Nodes Expanded   Max Queue Size   Time (s)
---------------------------------------------------------------------------
A* with Misplaced Tile     16      649              419              0.0105
A* with Manhattan Distance 16       96               64              0.0020
Uniform Cost Search        16     13653            7261              0.1768

```


## License
MIT License