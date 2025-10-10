import heapq
import copy
import time
import random
import os

# Global variable for goal state, will be set in main
GOAL_STATE = None
N = 0  # Dimension of the puzzle


# Utilities
def flatten(state):
    return tuple(num for row in state for num in row)


def find_blank(state):
    n_local = len(state)  # Get N from the state
    for i, row in enumerate(state):
        if 0 in row:
            return i, row.index(0)
    return -1, -1  # Should not happen for valid puzzles


# Calculate the target row and column for a given value in an N*N puzzle
def goal_position(value, n_local):
    if value == 0: return (n_local - 1, n_local - 1)  # Blank at the end
    value -= 1  # Adjust the value to be 0-based index
    return value // n_local, value % n_local


# Heuristics need N and the goal state
def misplaced_tiles(state, n_local, goal_state_local):
    count = 0
    for i in range(n_local):
        for j in range(n_local):
            if state[i][j] != 0 and state[i][j] != goal_state_local[i][j]:
                count += 1
    return count


def manhattan_distance(state, n_local,
                       goal_state_local):  # goal_state_local is not needed here but kept for consistency
    distance = 0
    for i in range(n_local):
        for j in range(n_local):
            val = state[i][j]
            if val != 0:
                gi, gj = goal_position(val, n_local)
                distance += abs(i - gi) + abs(j - gj)
    return distance


# Function to check if an N*N puzzle state is solvable
def is_solvable(state, n_local):
    flat_state = [num for row in state for num in row if num != 0]  # Ignore the blank tile (0)
    inversions = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1

    # For N*N puzzles:
    # If N is odd, the puzzle is solvable if the number of inversions is even.
    # If N is even, the puzzle is solvable if:
    #   - the blank is on an even row counting from the bottom (0-indexed) and inversions is odd.
    #   - the blank is on an odd row counting from the bottom (0-indexed) and inversions is even.
    if n_local % 2 != 0:  # N is odd (e.g., 3x3)
        return inversions % 2 == 0
    else:  # N is even (e.g., 4x4)
        blank_row, _ = find_blank(state)
        blank_row_from_bottom = n_local - 1 - blank_row
        if blank_row_from_bottom % 2 == 0:  # Blank on even row from bottom
            return inversions % 2 != 0
        else:  # Blank on odd row from bottom
            return inversions % 2 == 0


# Function to generate a random solvable N*N puzzle
def generate_random_puzzle(n_local):
    nums = list(range(n_local * n_local))
    while True:
        random.shuffle(nums)
        puzzle = []
        for i in range(0, n_local * n_local, n_local):
            puzzle.append(nums[i:i + n_local])
        if is_solvable(puzzle, n_local):
            return puzzle


# Node structure
class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.total = cost + heuristic

    def __lt__(self, other):
        return self.total < other.total

    def trace_path(self):
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]


# General Search Algorithm needs N and goal state for heuristics and goal check
def general_search(initial_state, heuristic_fn, n_local, goal_state_local):
    frontier = []
    # Pass N and goal state to heuristic function
    start_heuristic = heuristic_fn(initial_state, n_local, goal_state_local)
    heapq.heappush(frontier, Node(initial_state, cost=0, heuristic=start_heuristic))

    visited = set()
    max_queue_size = 0
    nodes_expanded = 0

    # Track the number of nodes at each depth
    depth_nodes_count = {}  # {depth: nodes_expanded_count}

    while frontier:
        max_queue_size = max(max_queue_size, len(frontier))
        node = heapq.heappop(frontier)
        flat = flatten(node.state)
        if flat in visited:
            continue
        visited.add(flat)
        nodes_expanded += 1

        # Record the number of nodes expanded at the current depth
        current_depth = node.cost
        if current_depth not in depth_nodes_count:
            depth_nodes_count[current_depth] = 0
        depth_nodes_count[current_depth] += 1

        # Compare with the dynamically generated goal state
        if node.state == goal_state_local:
            return {
                'solution_path': node.trace_path(),
                'depth': node.cost,
                'nodes_expanded': nodes_expanded,
                'max_queue_size': max_queue_size,
                'depth_nodes_count': depth_nodes_count
            }

        # Pass N to expand_node
        for move in expand_node(node.state, n_local):
            new_state = move
            if flatten(new_state) not in visited:
                # Pass N and goal state to heuristic function
                new_heuristic = heuristic_fn(new_state, n_local, goal_state_local)
                heapq.heappush(frontier,
                               Node(new_state, parent=node, cost=node.cost + 1, heuristic=new_heuristic))

    return None  # Failure


# Pass N to Generate new states
def expand_node(state, n_local):
    moves = []
    i, j = find_blank(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for di, dj in directions:
        ni, nj = i + di, j + dj
        # Check bounds using n_local
        if 0 <= ni < n_local and 0 <= nj < n_local:
            new_state = copy.deepcopy(state)
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            moves.append(new_state)
    return moves


# Helper function to print the puzzle state
def print_state(state, title=""):
    if title:
        print(title)
    for row in state:
        # Replace 0 with __ for better visibility, ensure numbers align
        print(" ".join(map(lambda x: str(x).rjust(2) if x != 0 else '__', row)))
    print()


# Function to generate the goal state for a given N
def generate_goal_state(n_local):
    goal = [[0] * n_local for _ in range(n_local)]
    num = 1
    for r in range(n_local):
        for c in range(n_local):
            goal[r][c] = num
            num += 1
    goal[n_local - 1][n_local - 1] = 0  # Set the last element to blank (0)
    return goal


# Read input and execute main program logic
def main():
    global GOAL_STATE, N  # Use global variables

    print("Welcome to the N-Puzzle Solver!")

    while True:
        try:
            n_input = input("Enter the puzzle dimension N (e.g., 3 for 8-puzzle, 4 for 15-puzzle): ").strip()
            N = int(n_input)
            if N < 2:
                raise ValueError("N must be at least 2.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter an integer >= 2.")

    # Generate the goal state based on N
    GOAL_STATE = generate_goal_state(N)
    total_numbers = N * N

    mode = input(f"Type '1' to use a randomly generated {N * N - 1}-puzzle, or '2' to create your own: ")

    initial = None  # Initialize initial state

    if mode == '1':
        # Generate a random solvable puzzle for the chosen N
        print(f"Generating a random solvable {N * N - 1}-puzzle...")
        initial = generate_random_puzzle(N)
        print("Using generated puzzle:")
        for r in initial: print(r)

    elif mode == '2':
        initial = []
        print(f"Enter the puzzle numbers (0 for blank), {N} numbers per row, row by row:")
        all_nums_expected = set(range(total_numbers))  # Expected numbers 0 to N*N-1
        entered_nums = set() # Deduplicated set

        for i in range(N):
            while True:
                try:
                    row_str = input(f"Enter row {i + 1} ({N} space-separated numbers): ").strip()
                    row = list(map(int, row_str.split()))
                    if len(row) != N:
                        raise ValueError(f"Please enter exactly {N} numbers.")

                    current_row_nums = set()
                    for num in row:
                        if not (0 <= num < total_numbers):
                            raise ValueError(f"Numbers must be between 0 and {total_numbers - 1}.")
                        if num in current_row_nums:
                            raise ValueError(f"Duplicate number {num} in the row.")
                        if num in entered_nums:
                            raise ValueError(f"Duplicate number {num} from previous rows.")
                        current_row_nums.add(num)

                    initial.append(row)
                    entered_nums.update(current_row_nums)
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")

        if entered_nums != all_nums_expected:
            missing = all_nums_expected - entered_nums
            extra = entered_nums - all_nums_expected
            error_msg = "Error: The final puzzle configuration is invalid."
            if missing:
                error_msg += f" Missing numbers: {sorted(list(missing))}."
            if extra:
                error_msg += f" Unexpected/Duplicate numbers: {sorted(list(extra))}."
            print(error_msg)
            if not is_solvable(initial, N):
                print("Error: The entered puzzle configuration is also not solvable.")
            return

        if not is_solvable(initial, N):
            print("\nError: The entered puzzle configuration is not solvable.")
            return

    else:
        print("Invalid mode selected. Please run the script again and choose '1' or '2'.")
        return

    if initial is None:
        print("No puzzle state defined. Exiting.")
        return

    print_state(initial, f"\nInitial {N * N - 1}-Puzzle State:")

    algorithms = {
        "A* with Misplaced Tile": misplaced_tiles,
        "A* with Manhattan Distance": manhattan_distance,

        # A 'heuristic function' that always returns 0, making A* search degenerate into UCS
        "Uniform Cost Search": lambda s, n_local, gs: 0
    }

    results = {}
    solution_found_by_any = False

    print("\nRunning algorithms...")
    for name, heuristic in algorithms.items():
        print(f"--- {name} ---")
        start_time = time.time()
        result = general_search(copy.deepcopy(initial), heuristic, N, GOAL_STATE)
        end_time = time.time()
        execution_time = end_time - start_time

        if result:
            solution_found_by_any = True
            results[name] = {
                'depth': result['depth'],
                'nodes_expanded': result['nodes_expanded'],
                'max_queue_size': result['max_queue_size'],
                'execution_time': execution_time,
                'solution_found': True,
                'depth_nodes_count': result['depth_nodes_count']  # Save depth-nodes count
            }
            print(f"  Solution found!")

            # Create output directory if it doesn't exist
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # Save depth-nodes count to file
            algorithm_filename = name.replace(' ', '_').replace('*', 'Star')
            file_path = os.path.join(output_dir, f"{algorithm_filename}_depth_nodes.txt")

            with open(file_path, 'w') as f:
                f.write(f"Depth-Nodes Count for {name}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Initial state:\n")
                for row in initial:
                    f.write(f"{row}\n")
                f.write("\n")
                f.write(f"Solution depth: {result['depth']}\n")
                f.write(f"Total nodes expanded: {result['nodes_expanded']}\n")
                f.write(f"Max queue size: {result['max_queue_size']}\n")
                f.write(f"Execution time: {execution_time:.4f} seconds\n\n")
                f.write("Depth-by-depth node counts:\n")
                f.write("-" * 40 + "\n")
                for depth, count in sorted(result['depth_nodes_count'].items()):
                    f.write(f"Depth {depth}: {count} nodes\n")

            print(f"  Depth-nodes data saved to {file_path}")

            # Print the solution path for this algorithm
            print(f"  Solution Path ({name}):")
            solution_path = result['solution_path']
            for i, step_state in enumerate(solution_path):
                print_state(step_state, f"  Step {i}:")
            print("-" * 80)

        else:
            results[name] = {
                'depth': 'N/A',
                'nodes_expanded': 'N/A',
                'max_queue_size': 'N/A',
                'execution_time': execution_time,
                'solution_found': False
            }
            print("  No solution found (or ran out of memory/time).")

    if solution_found_by_any:
        print_state(GOAL_STATE, f"\nGoal {N * N - 1}-Puzzle State:")

    print("\n--- Algorithm Comparison ---")
    print(f"{'Algorithm':<30} | {'Depth':<8} | {'Nodes Expanded':<15} | {'Max Queue Size':<15} | {'Time (s)':<10}")
    print("-" * 85)
    for name, data in results.items():
        depth_str = str(data['depth']) if data['solution_found'] else 'N/A'
        nodes_str = str(data['nodes_expanded']) if data['solution_found'] else 'N/A'
        queue_str = str(data['max_queue_size']) if data['solution_found'] else 'N/A'
        time_str = f"{data['execution_time']:.4f}"
        print(f"{name:<30} | {depth_str:<8} | {nodes_str:<15} | {queue_str:<15} | {time_str:<10}")


if __name__ == "__main__":
    main()
