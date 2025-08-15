import math
import random
import time
import os
import statistics
from typing import List, Tuple, Set, Optional, Dict, Any

class ConstrainedTSP:
    """
    Constrained TSP problem class that handles the modified TSP with forbidden edge constraints.
    """
    
    def __init__(self, tsp_file: str, opt_file: str, delta: int):
        """Initialize the constrained TSP problem."""
        self.tsp_file = tsp_file
        self.opt_file = opt_file
        self.nodes = {}
        self.distances = {}
        self.n = 0
        self.optimal_tour = []
        self.optimal_length = 0
        self.delta = delta
        self.forbidden_pairs = set()
        self.edge_weight_type = "EUC_2D"
        
        self._read_tsp_file(tsp_file)
        self._read_opt_file(opt_file)
        self._calculate_distances()
        self._generate_constraints()
        
        if self.optimal_tour:
            calculated_length = self.calculate_tour_length(self.optimal_tour)
            if abs(calculated_length - self.optimal_length) > self.optimal_length * 0.1:
                print(f"Warning: Calculated length {calculated_length} differs from stated optimal {self.optimal_length}")
                self.optimal_length = calculated_length
    
    def _read_tsp_file(self, filename: str) -> None:
        """Read TSP instance file and extract node coordinates."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"TSP file '{filename}' not found!")
        
        coord_section = False
        for line in lines:
            line = line.strip()
            if line.startswith('DIMENSION'):
                self.n = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                self.edge_weight_type = line.split(':')[1].strip()
            elif line == 'NODE_COORD_SECTION':
                coord_section = True
            elif coord_section and line != 'EOF' and line:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.nodes[node_id] = (x, y)
    
    def _read_opt_file(self, filename: str) -> None:
        """Read optimal tour file and extract tour sequence and length."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"OPT tour file '{filename}' not found!")
        
        tour_section = False
        for line in lines:
            line = line.strip()
            if 'Length' in line or 'length' in line or 'COMMENT' in line:
                import re
                match = re.search(r'\(.*?(\d+).*?\)', line)
                if not match:
                    match = re.search(r'(\d+)', line)
                if match:
                    self.optimal_length = int(match.group(1))
            elif line == 'TOUR_SECTION':
                tour_section = True
            elif tour_section and line not in ['EOF', '-1', '']:
                if line.isdigit():
                    self.optimal_tour.append(int(line))
        
        if not self.optimal_tour:
            raise ValueError("No valid tour found in opt file!")
    
    def _geo_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate geographical distance between two coordinates."""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        PI = 3.141592653589793
        deg_to_rad = PI / 180.0
        
        lat1_deg = int(lat1) + (lat1 - int(lat1)) * 100 / 60
        lon1_deg = int(lon1) + (lon1 - int(lon1)) * 100 / 60
        lat2_deg = int(lat2) + (lat2 - int(lat2)) * 100 / 60
        lon2_deg = int(lon2) + (lon2 - int(lon2)) * 100 / 60
        
        lat1_rad = lat1_deg * deg_to_rad
        lon1_rad = lon1_deg * deg_to_rad
        lat2_rad = lat2_deg * deg_to_rad
        lon2_rad = lon2_deg * deg_to_rad
        
        R = 6378.388
        
        q1 = math.cos(lon1_rad - lon2_rad)
        q2 = math.cos(lat1_rad - lat2_rad)
        q3 = math.cos(lat1_rad + lat2_rad)
        
        distance = R * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3))
        return int(distance + 1.0)
    
    def _euc_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two coordinates."""
        x1, y1 = coord1
        x2, y2 = coord2
        return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    
    def _calculate_distances(self) -> None:
        """Calculate distance matrix for all node pairs."""
        for i in range(1, self.n + 1):
            for j in range(i + 1, self.n + 1):
                if i in self.nodes and j in self.nodes:
                    if self.edge_weight_type == "GEO":
                        dist = self._geo_distance(self.nodes[i], self.nodes[j])
                    else:
                        dist = self._euc_distance(self.nodes[i], self.nodes[j])
                    
                    self.distances[(i, j)] = dist
                    self.distances[(j, i)] = dist
        
        for i in range(1, self.n + 1):
            self.distances[(i, i)] = 0
    
    def _generate_constraints(self) -> None:
        """Generate forbidden edge constraints based on optimal tour and delta parameter."""
        tour = self.optimal_tour
        n = len(tour)
        
        for i in range(n):
            current_node = tour[i]
            forbidden_nodes = set()
            
            for j in range(max(0, i - self.delta), i):
                forbidden_nodes.add(tour[j])
            
            for j in range(i + 1, min(n, i + 1 + self.delta)):
                forbidden_nodes.add(tour[j])
            
            for forbidden_node in forbidden_nodes:
                self.forbidden_pairs.add((current_node, forbidden_node))
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate total length of a tour."""
        length = 0
        for i in range(len(tour)):
            current = tour[i]
            next_node = tour[(i + 1) % len(tour)]
            if (current, next_node) in self.distances:
                length += self.distances[(current, next_node)]
        return length
    
    def calculate_infeasibility(self, tour: List[int]) -> int:
        """Calculate number of constraint violations in a tour."""
        violations = 0
        for i in range(len(tour)):
            current = tour[i]
            next_node = tour[(i + 1) % len(tour)]
            if (current, next_node) in self.forbidden_pairs:
                violations += 1
        return violations

class ProgressiveConstraintRelaxationILS:
    """
    Progressive Constraint Relaxation Iterated Local Search algorithm.
    """
    
    def __init__(self, problem: ConstrainedTSP, seed: int = 0):
        """Initialize the algorithm with problem instance and parameters."""
        self.problem = problem
        self.pareto_solutions = []  # List of (tour, (length, violations)) tuples
        
        # Algorithm parameters
        self.max_pareto_size = 25
        self.lambda_penalty = 1.0
        self.lambda_increase_rate = 1.3
        self.lambda_decrease_rate = 0.85
        
        # Algorithm state variables
        self.start_time = 0
        self.log_file = None
        self.feasible_found = False
        self.feasible_mode = False
        self.iteration_count = 0
        
        random.seed(seed)
        
    def _canonical_tour(self, tour: List[int]) -> tuple:
        """Convert tour to canonical form to avoid duplicate solutions."""
        n = len(tour)
        
        min_idx = min(range(n), key=lambda i: tour[i])
        rotated_forward = tour[min_idx:] + tour[:min_idx]
        
        reversed_tour = list(reversed(tour))
        min_idx_rev = min(range(n), key=lambda i: reversed_tour[i])
        rotated_reversed = reversed_tour[min_idx_rev:] + reversed_tour[:min_idx_rev]
        
        return tuple(min(rotated_forward, rotated_reversed))
    
    def _is_dominated(self, sol1: Tuple[float, int], sol2: Tuple[float, int]) -> bool:
        """Check if solution sol2 dominates solution sol1 in Pareto sense."""
        length1, infeas1 = sol1
        length2, infeas2 = sol2
        return (length2 <= length1 and infeas2 <= infeas1) and (length2 < length1 or infeas2 < infeas1)
    
    def _update_pareto_set(self, solution: List[int]) -> bool:
        """Update Pareto set with new solution if it's non-dominated."""
        length = self.problem.calculate_tour_length(solution)
        infeas = self.problem.calculate_infeasibility(solution)
        new_obj = (length, infeas)
        
        if not self.feasible_found and infeas == 0:
            self.feasible_found = True
            self.feasible_mode = True
            self.lambda_penalty = 1.0
            print(f"# First feasible solution found at iteration {self.iteration_count}")
        
        canonical_new = self._canonical_tour(solution)
        for sol, obj in self.pareto_solutions:
            if self._canonical_tour(sol) == canonical_new:
                return False
        
        dominated = False
        for sol, obj in self.pareto_solutions:
            if self._is_dominated(new_obj, obj):
                dominated = True
                break
        
        if not dominated:
            self.pareto_solutions = [(sol, obj) for sol, obj in self.pareto_solutions 
                                   if not self._is_dominated(obj, new_obj)]
            self.pareto_solutions.append((solution.copy(), new_obj))
            
            if len(self.pareto_solutions) > self.max_pareto_size:
                
                def sort_key(x):
                    tour, (length, violations) = x
                    return (violations, length)  
                
                self.pareto_solutions.sort(key=sort_key)
                trimmed = []
                for sol, obj in self.pareto_solutions:
                    if obj[1] <= 3 or len(trimmed) < self.max_pareto_size:
                        trimmed.append((sol, obj))
                self.pareto_solutions = trimmed[:self.max_pareto_size]
            
            return True
        
        return False
    
    def _calculate_objective_value(self, length: float, infeasibility: int) -> float:
        """Calculate objective function value based on current optimization mode."""
        if self.feasible_mode and infeasibility == 0:
            return length
        else:
            penalty_weight = self.lambda_penalty * self.problem.optimal_length / self.problem.n
            return length + penalty_weight * infeasibility
    
    def _print_solution(self, solution: List[int], is_new: bool = False) -> None:
        """Print solution in required format: sequence objective_value infeasibility_degree time"""
        if not is_new:
            return
            
        length = self.problem.calculate_tour_length(solution)
        infeas = self.problem.calculate_infeasibility(solution)
        current_time = time.time() - self.start_time
        
        objective_f = self._calculate_objective_value(length, infeas)
        
        tour_sequence = '-'.join(map(str, solution))
        print(f"{tour_sequence} {objective_f:.2f} {infeas} {current_time:.2f}")
        
        if self.log_file:
            self.log_file.write(f"{tour_sequence} {objective_f:.2f} {infeas} {current_time:.2f}\n")
            self.log_file.flush()
    
    def _two_opt_feasible_enhanced(self, tour: List[int], max_iterations: int = 3) -> List[int]:
        """Enhanced 2-opt local search with feasibility preservation."""
        n = len(tour)
        current = tour[:]
        
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(n - 1):
                for j in range(i + 2, n - (1 if i == 0 else 0)):
                    a, b = current[i], current[(i + 1) % n]
                    c, d = current[j], current[(j + 1) % n]
                    
                    if (a, c) in self.problem.forbidden_pairs:
                        continue
                    if (b, d) in self.problem.forbidden_pairs:
                        continue
                    
                    new_tour = current[:i+1] + list(reversed(current[i+1:j+1])) + current[j+1:]
                    
                    new_infeas = self.problem.calculate_infeasibility(new_tour)
                    curr_infeas = self.problem.calculate_infeasibility(current)
                    
                    if (new_infeas <= curr_infeas and 
                        self.problem.calculate_tour_length(new_tour) < self.problem.calculate_tour_length(current)):
                        current = new_tour
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    def _or_opt_feasible(self, tour: List[int], segment_lengths: List[int] = [1, 2, 3]) -> List[int]:
        """Or-opt local search with feasibility preservation."""
        n = len(tour)
        current = tour[:]
        improved = True
        
        while improved:
            improved = False
            
            for seg_len in segment_lengths:
                if seg_len >= n:
                    continue
                    
                for i in range(n - seg_len + 1):
                    segment = current[i:i+seg_len]
                    remaining = current[:i] + current[i+seg_len:]
                    
                    for pos in range(len(remaining) + 1):
                        if pos < len(remaining):
                            prev_node = remaining[pos-1] if pos > 0 else remaining[-1]
                            next_node = remaining[pos]
                        else:
                            prev_node = remaining[-1]
                            next_node = remaining[0]
                        
                        if (prev_node, segment[0]) in self.problem.forbidden_pairs:
                            continue
                        if (segment[-1], next_node) in self.problem.forbidden_pairs:
                            continue
                        
                        candidate = remaining[:pos] + segment + remaining[pos:]
                        
                        if (self.problem.calculate_infeasibility(candidate) <= self.problem.calculate_infeasibility(current) and
                            self.problem.calculate_tour_length(candidate) < self.problem.calculate_tour_length(current)):
                            current = candidate
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return current
    
    def _double_bridge_perturbation(self, tour: List[int]) -> List[int]:
        """Apply double-bridge perturbation to escape local optima."""
        n = len(tour)
        if n < 8:
            return tour[:]
        
        positions = sorted(random.sample(range(n), 4))
        a, b, c, d = positions
        
        seg_A = tour[:a]
        seg_B = tour[a:b]
        seg_C = tour[b:c]
        seg_D = tour[c:d]
        seg_E = tour[d:]
        
        return seg_A + seg_D + seg_C + seg_B + seg_E
    
    def _repair_shortest_infeasible(self) -> Optional[List[int]]:
        """Attempt to repair the shortest infeasible solution by relocating violating nodes."""
        infeasible_solutions = [(tour, obj) for (tour, obj) in self.pareto_solutions if obj[1] > 0]
        if not infeasible_solutions:
            return None
        
      
        best_tour, (best_length, best_violations) = min(infeasible_solutions, 
                                                       key=lambda x: x[1][0])  
        current = best_tour[:]
        n = len(current)
        
        for attempt in range(min(100, 3 * n)):
            violating_edges = []
            for i in range(n):
                current_node = current[i]
                next_node = current[(i + 1) % n]
                if (current_node, next_node) in self.problem.forbidden_pairs:
                    violating_edges.append(i)
            
            if not violating_edges:
                break
            
            edge_idx = random.choice(violating_edges)
            node_to_relocate_idx = (edge_idx + 1) % n
            node_to_relocate = current[node_to_relocate_idx]
            
            temp_tour = current[:node_to_relocate_idx] + current[node_to_relocate_idx+1:]
            
            best_position = None
            best_score = (float('inf'), float('inf'))  # (violations, length_change)
            
            for pos in range(len(temp_tour)):
                prev_node = temp_tour[pos-1] if pos > 0 else temp_tour[-1]
                next_node = temp_tour[pos] if pos < len(temp_tour) else temp_tour[0]
                
                if ((prev_node, node_to_relocate) in self.problem.forbidden_pairs or
                    (node_to_relocate, next_node) in self.problem.forbidden_pairs):
                    continue
                
                candidate = temp_tour[:pos] + [node_to_relocate] + temp_tour[pos:]
                new_violations = self.problem.calculate_infeasibility(candidate)
                new_length = self.problem.calculate_tour_length(candidate)
                
                score = (new_violations, new_length - best_length)
                if score < best_score:
                    best_score = score
                    best_position = pos
            
            if best_position is not None:
                current = temp_tour[:best_position] + [node_to_relocate] + temp_tour[best_position:]
                current = self._or_opt_feasible(current, [1, 2])
                current = self._two_opt_feasible_enhanced(current, max_iterations=1)
            else:
                current = self._double_bridge_perturbation(current)
        
        return current
    
    def _constraint_aware_construction(self) -> List[int]:
        """Construct initial solution using greedy randomized approach with constraint awareness."""
        best_tour = None
        best_score = float('inf')
        
        for attempt in range(5):
            unvisited = set(range(1, self.problem.n + 1))
            current_node = random.choice(list(unvisited))
            tour = [current_node]
            unvisited.remove(current_node)
            
            while unvisited:
                candidates = []
                for next_node in unvisited:
                    distance = self.problem.distances.get((current_node, next_node), float('inf'))
                    
                    penalty = 0
                    if (current_node, next_node) in self.problem.forbidden_pairs:
                        penalty = self.problem.optimal_length / self.problem.n * 2
                    
                    candidates.append((next_node, distance + penalty))
                
                candidates.sort(key=lambda x: x[1])
                alpha = 0.25
                best_cost = candidates[0][1]
                worst_cost = candidates[-1][1]
                threshold = best_cost + alpha * (worst_cost - best_cost)
                rcl = [c for c in candidates if c[1] <= threshold]
                
                next_node = random.choice(rcl)[0]
                tour.append(next_node)
                unvisited.remove(next_node)
                current_node = next_node
            
            length = self.problem.calculate_tour_length(tour)
            violations = self.problem.calculate_infeasibility(tour)
            score = self._calculate_objective_value(length, violations)
            
            if score < best_score:
                best_tour = tour
                best_score = score
        
        return best_tour or list(range(1, self.problem.n + 1))
    
    def _adaptive_lambda_update(self) -> None:
        """Update penalty parameter based on current Pareto set composition."""
        if not self.pareto_solutions or self.feasible_found:
            return
        
        feasible_count = len([s for s in self.pareto_solutions if s[1][1] == 0])
        total_solutions = len(self.pareto_solutions)
        
        if feasible_count == 0:
            old_lambda = self.lambda_penalty
            self.lambda_penalty *= self.lambda_increase_rate
            if self.log_file:
                self.log_file.write(f"# Lambda increased: {old_lambda:.3f} -> {self.lambda_penalty:.3f}\n")
        elif feasible_count > total_solutions * 0.6:
            old_lambda = self.lambda_penalty
            self.lambda_penalty *= self.lambda_decrease_rate
            if self.log_file:
                self.log_file.write(f"# Lambda decreased: {old_lambda:.3f} -> {self.lambda_penalty:.3f}\n")
    
    def _select_current_solution(self) -> List[int]:
        """Select current solution for optimization based on algorithm mode."""
        if not self.pareto_solutions:
            return self._constraint_aware_construction()
        
        if self.feasible_mode:
            # Prioritize feasible solutions
            feasible_solutions = [(tour, obj) for tour, obj in self.pareto_solutions if obj[1] == 0]
            if feasible_solutions:
                # Select best feasible solution by length
                selected = min(feasible_solutions, key=lambda x: x[1][0])
                return selected[0].copy()
            else:
                # Select best infeasible solution by violations then length
                selected = min(self.pareto_solutions, key=lambda x: (x[1][1], x[1][0]))
                return selected[0].copy()
        else:
            # Use penalty function
            penalty_weight = self.lambda_penalty * self.problem.optimal_length / self.problem.n
            
            def penalty_score(x):
                tour, (length, violations) = x
                return length + penalty_weight * violations
            
            selected = min(self.pareto_solutions, key=penalty_score)
            return selected[0].copy()
    
    def solve(self, time_limit: int = 120, log_filename: Optional[str] = None) -> List[Tuple[List[int], Tuple[float, int]]]:
        """Main solving method implementing the Progressive Constraint Relaxation ILS algorithm."""
        self.start_time = time.time()
        
        if log_filename:
            self.log_file = open(log_filename, 'w')
            self.log_file.write(f"Progressive Constraint Relaxation ILS Log\n")
            self.log_file.write(f"Instance: {self.problem.tsp_file}, Delta: {self.problem.delta}\n")
            self.log_file.write(f"Nodes: {self.problem.n}, Optimal: {self.problem.optimal_length}\n")
            self.log_file.write(f"Constraints: {len(self.problem.forbidden_pairs)}\n\n")
        
        # Phase 1: Initialize with original optimal solution (infeasible)
        current_solution = self.problem.optimal_tour.copy()
        is_new = self._update_pareto_set(current_solution)
        self._print_solution(current_solution, is_new)
        
        # Phase 2: Generate diverse initial solutions
        for attempt in range(3):
            constructed_solution = self._constraint_aware_construction()
            is_new = self._update_pareto_set(constructed_solution)
            self._print_solution(constructed_solution, is_new)
        
        # Phase 3: Main optimization loop
        self.iteration_count = 0
        last_lambda_update = 0
        last_repair_attempt = 0
        
        while time.time() - self.start_time < time_limit:
            self.iteration_count += 1
            
            # Select current solution using fixed method
            current_solution = self._select_current_solution()
            
            # Apply optimization operators
            if self.feasible_mode and self.problem.calculate_infeasibility(current_solution) == 0:
                # Intensive optimization for feasible solutions
                if self.iteration_count % 15 == 0:
                    perturbed = self._double_bridge_perturbation(current_solution)
                    optimized = self._two_opt_feasible_enhanced(perturbed, max_iterations=3)
                    optimized = self._or_opt_feasible(optimized, [1, 2, 3])
                    optimized = self._two_opt_feasible_enhanced(optimized, max_iterations=2)
                    
                    is_new = self._update_pareto_set(optimized)
                    self._print_solution(optimized, is_new)
                else:
                    optimized = self._two_opt_feasible_enhanced(current_solution, max_iterations=2)
                    optimized = self._or_opt_feasible(optimized, [1, 2, 3])
                    optimized = self._two_opt_feasible_enhanced(optimized, max_iterations=1)
                    
                    is_new = self._update_pareto_set(optimized)
                    self._print_solution(optimized, is_new)
            else:
                improved = self._two_opt_feasible_enhanced(current_solution, max_iterations=1)
                is_new = self._update_pareto_set(improved)
                self._print_solution(improved, is_new)
            
            # Periodic repair mechanism
            if self.iteration_count - last_repair_attempt >= 25:
                repaired = self._repair_shortest_infeasible()
                if repaired:
                    is_new = self._update_pareto_set(repaired)
                    self._print_solution(repaired, is_new)
                last_repair_attempt = self.iteration_count
            
            # Adaptive parameter management
            if not self.feasible_found and self.iteration_count - last_lambda_update >= 30:
                self._adaptive_lambda_update()
                last_lambda_update = self.iteration_count
            
            # Progress reporting
            if self.iteration_count % 100 == 0 and self.log_file:
                feasible_count = len([s for s in self.pareto_solutions if s[1][1] == 0])
                elapsed = time.time() - self.start_time
                self.log_file.write(f"# Iteration {self.iteration_count}, Time {elapsed:.1f}s, "
                                  f"Pareto size: {len(self.pareto_solutions)}, "
                                  f"Feasible: {feasible_count}\n")
        
        # Cleanup
        if self.log_file:
            final_stats = self._get_final_statistics()
            self.log_file.write(f"\n# Final Statistics:\n")
            for key, value in final_stats.items():
                self.log_file.write(f"# {key}: {value}\n")
            self.log_file.close()
        
        return self.pareto_solutions
    
    def _get_final_statistics(self) -> Dict[str, Any]:
        """Generate final algorithm statistics."""
        feasible_solutions = [s for s in self.pareto_solutions if s[1][1] == 0]
        
        stats = {
            'Total iterations': self.iteration_count,
            'Total solutions found': len(self.pareto_solutions),
            'Feasible solutions': len(feasible_solutions),
            'Feasibility rate': f"{len(feasible_solutions)/len(self.pareto_solutions)*100:.1f}%" if self.pareto_solutions else "0%",
            'Best feasible length': min([s[1][0] for s in feasible_solutions]) if feasible_solutions else "N/A",
            'Best feasible gap': f"{(min([s[1][0] for s in feasible_solutions]) - self.problem.optimal_length) / self.problem.optimal_length * 100:.2f}%" if feasible_solutions else "N/A",
            'Min violations': min([s[1][1] for s in self.pareto_solutions]) if self.pareto_solutions else "N/A"
        }
        
        return stats

def auto_find_opt_file(tsp_file: str) -> str:
    """Automatically find the corresponding optimal tour file for a TSP instance."""
    base_name = os.path.splitext(tsp_file)[0]
    
    possible_extensions = [".opt.tour", ".tour", ".opt", ".sol"]
    
    for ext in possible_extensions:
        opt_file = base_name + ext
        if os.path.exists(opt_file):
            return opt_file
    
    raise FileNotFoundError(f"Could not find optimal tour file for {tsp_file}. "
                          f"Tried: {[base_name + ext for ext in possible_extensions]}")

def interactive_input() -> Tuple[str, str, int, int, int, Optional[int]]:
    """Interactive input collection for algorithm parameters."""
    print("=" * 70)
    print("Progressive Constraint Relaxation ILS for Constrained TSP")
    print("=" * 70)
    
    tsp_file = input("Enter TSP instance filename (e.g., berlin52.tsp): ").strip()
    if not os.path.exists(tsp_file):
        print(f"Warning: File '{tsp_file}' not found in current directory.")
    
    try:
        opt_file = auto_find_opt_file(tsp_file)
        print(f"Found optimal tour file: {opt_file}")
    except FileNotFoundError as e:
        print(str(e))
        opt_file = input("Please enter optimal tour filename manually: ").strip()
    
    while True:
        try:
            delta = int(input("Enter Delta value (1, 3, 5, or 8): ").strip())
            if delta in [1, 3, 5, 8]:
                break
            else:
                print("Delta must be one of: 1, 3, 5, 8")
        except ValueError:
            print("Please enter a valid integer.")
    
    try:
        time_limit_input = input("Enter time limit in seconds (default 120): ").strip()
        time_limit = int(time_limit_input) if time_limit_input else 120
    except ValueError:
        print("Invalid input, using default time limit: 120 seconds")
        time_limit = 120
    
    try:
        rep_input = input("Enter number of replications (default 5): ").strip()
        replications = int(rep_input) if rep_input else 5
    except ValueError:
        print("Invalid input, using default replications: 5")
        replications = 5
    
    seed_input = input("Enter custom seed (optional, press Enter for default): ").strip()
    custom_seed = int(seed_input) if seed_input.isdigit() else None
    
    return tsp_file, opt_file, delta, time_limit, replications, custom_seed

def run_single_experiment(tsp_file: str, opt_file: str, delta: int, seed: int, 
                         time_limit: int = 120) -> Dict[str, Any]:
    """Run a single experimental replication."""
    try:
        if not os.path.exists(tsp_file):
            raise FileNotFoundError(f"TSP file {tsp_file} not found")
        if not os.path.exists(opt_file):
            raise FileNotFoundError(f"OPT file {opt_file} not found")
        
        problem = ConstrainedTSP(tsp_file, opt_file, delta)
        solver = ProgressiveConstraintRelaxationILS(problem, seed)
        
        print(f"\n{'='*70}")
        print(f"Replication {seed+1}: {tsp_file}, Delta: {delta}")
        print(f"Nodes: {problem.n}, Optimal: {problem.optimal_length}, "
              f"Constraints: {len(problem.forbidden_pairs)}")
        print(f"Random seed: {seed}")
        print(f"{'='*70}")
        
        base_name = os.path.splitext(tsp_file)[0]
        log_filename = f"{base_name}_delta{delta}_rep{seed+1}.log"
        
        start_time = time.time()
        solutions = solver.solve(time_limit, log_filename)
        solve_time = time.time() - start_time
        
        if solutions:
            feasible_solutions = [s for s in solutions if s[1][1] == 0]
            if feasible_solutions:
                best_feasible = min(feasible_solutions, key=lambda x: x[1][0])
                gap = (best_feasible[1][0] - problem.optimal_length) / problem.optimal_length * 100
                return {
                    'status': 'success',
                    'solutions': len(solutions),
                    'feasible_count': len(feasible_solutions),
                    'best_gap': gap,
                    'min_violations': 0,
                    'solve_time': solve_time,
                    'best_length': best_feasible[1][0],
                    'avg_length': sum(s[1][0] for s in feasible_solutions) / len(feasible_solutions)
                }
            else:
                best_solution = min(solutions, key=lambda x: (x[1][1], x[1][0]))
                gap = (best_solution[1][0] - problem.optimal_length) / problem.optimal_length * 100
                return {
                    'status': 'no_feasible',
                    'solutions': len(solutions),
                    'feasible_count': 0,
                    'best_gap': gap,
                    'min_violations': best_solution[1][1],
                    'solve_time': solve_time,
                    'best_length': best_solution[1][0],
                    'avg_length': best_solution[1][0]
                }
        
        return {
            'status': 'failed',
            'solutions': 0,
            'feasible_count': 0,
            'best_gap': float('inf'),
            'min_violations': float('inf'),
            'solve_time': solve_time,
            'best_length': float('inf'),
            'avg_length': float('inf')
        }
    
    except Exception as e:
        print(f"Error in replication {seed+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e),
            'solutions': 0,
            'feasible_count': 0,
            'best_gap': float('inf'),
            'min_violations': float('inf'),
            'solve_time': 0,
            'best_length': float('inf'),
            'avg_length': float('inf')
        }

def analyze_and_report_results(results: List[Dict[str, Any]], tsp_file: str, 
                             delta: int, replications: int) -> None:
    """Analyze and report experimental results with comprehensive statistics."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENTAL ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Instance: {tsp_file}")
    print(f"Delta parameter: {delta}")
    print(f"Total replications: {replications}")
    
    successful_runs = [r for r in results if r['status'] in ['success', 'no_feasible']]
    error_runs = [r for r in results if r['status'] == 'error']
    
    print(f"Successful runs: {len(successful_runs)}/{replications}")
    print(f"Error runs: {len(error_runs)}")
    
    if error_runs:
        print("Errors encountered:")
        for i, error_run in enumerate(error_runs):
            print(f"  Run {i+1}: {error_run.get('error', 'Unknown error')}")
    
    if not successful_runs:
        print("No successful runs to analyze!")
        return
    
    feasible_runs = [r for r in successful_runs if r['feasible_count'] > 0]
    total_feasible_solutions = sum(r['feasible_count'] for r in successful_runs)
    
    print(f"\nFEASIBILITY ANALYSIS:")
    print(f"Replications with feasible solutions: {len(feasible_runs)}/{len(successful_runs)}")
    print(f"Success rate: {len(feasible_runs)/len(successful_runs)*100:.1f}%")
    print(f"Total feasible solutions found: {total_feasible_solutions}")
    
    print(f"\nPER-REPLICATION RESULTS:")
    for i, result in enumerate(results):
        if result['status'] == 'success':
            print(f"  Rep {i+1}: ✓ {result['feasible_count']} feasible, "
                  f"Best Length={result['best_length']:.0f}, "
                  f"Gap={result['best_gap']:.1f}%, Time={result['solve_time']:.1f}s")
        elif result['status'] == 'no_feasible':
            print(f"  Rep {i+1}: ✗ {result['feasible_count']} feasible, "
                  f"Min Violations={result['min_violations']}, "
                  f"Best Length={result['best_length']:.0f}, Time={result['solve_time']:.1f}s")
        else:
            print(f"  Rep {i+1}: ERROR - {result.get('error', 'Unknown error')}")
    
    if feasible_runs:
        feasible_gaps = [r['best_gap'] for r in feasible_runs]
        feasible_lengths = [r['best_length'] for r in feasible_runs]
        feasible_times = [r['solve_time'] for r in feasible_runs]
        
        print(f"\nFEASIBLE SOLUTIONS STATISTICS:")
        print(f"Gap Analysis:")
        print(f"  Best gap: {min(feasible_gaps):.1f}%")
        print(f"  Worst gap: {max(feasible_gaps):.1f}%")
        print(f"  Mean gap: {statistics.mean(feasible_gaps):.1f}%")
        print(f"  Median gap: {statistics.median(feasible_gaps):.1f}%")
        if len(feasible_gaps) > 1:
            print(f"  Std deviation: {statistics.stdev(feasible_gaps):.1f}%")
        
        print(f"Length Analysis:")
        print(f"  Best length: {min(feasible_lengths):.0f}")
        print(f"  Worst length: {max(feasible_lengths):.0f}")
        print(f"  Mean length: {statistics.mean(feasible_lengths):.0f}")
        print(f"  Median length: {statistics.median(feasible_lengths):.0f}")
        
        print(f"Runtime Analysis:")
        print(f"  Min time: {min(feasible_times):.1f}s")
        print(f"  Max time: {max(feasible_times):.1f}s")
        print(f"  Mean time: {statistics.mean(feasible_times):.1f}s")
    
    infeasible_runs = [r for r in successful_runs if r['feasible_count'] == 0]
    if infeasible_runs:
        min_violations = [r['min_violations'] for r in infeasible_runs]
        infeasible_lengths = [r['best_length'] for r in infeasible_runs]
        
        print(f"\nINFEASIBLE SOLUTIONS ANALYSIS:")
        print(f"Constraint Violations:")
        print(f"  Best (min violations): {min(min_violations)}")
        print(f"  Worst (max violations): {max(min_violations)}")
        print(f"  Mean violations: {statistics.mean(min_violations):.1f}")
        
        print(f"Length Analysis (infeasible):")
        print(f"  Best length: {min(infeasible_lengths):.0f}")
        print(f"  Mean length: {statistics.mean(infeasible_lengths):.0f}")
    
    all_times = [r['solve_time'] for r in successful_runs]
    print(f"\nOVERALL PERFORMANCE SUMMARY:")
    print(f"Runtime Statistics:")
    print(f"  Total runtime: {sum(all_times):.1f}s")
    print(f"  Average per replication: {statistics.mean(all_times):.1f}s")
    print(f"  Runtime std deviation: {statistics.stdev(all_times):.1f}s" if len(all_times) > 1 else "")
    
    print(f"\nALGORITHM EFFECTIVENESS:")
    if len(feasible_runs) >= len(successful_runs) * 0.8:
        print("✓ Excellent feasibility performance (≥80% success rate)")
    elif len(feasible_runs) >= len(successful_runs) * 0.6:
        print("◐ Good feasibility performance (≥60% success rate)")
    else:
        print("✗ Limited feasibility performance (<60% success rate)")
    
    if feasible_runs:
        avg_gap = statistics.mean([r['best_gap'] for r in feasible_runs])
        if avg_gap <= 50:
            print("✓ Excellent solution quality (≤50% gap)")
        elif avg_gap <= 100:
            print("◐ Good solution quality (≤100% gap)")
        else:
            print("✗ Limited solution quality (>100% gap)")
    
    print(f"{'='*80}")

def main():
    """Main execution function with comprehensive experiment management."""
    try:
        tsp_file, opt_file, delta, time_limit, replications, custom_seed = interactive_input()
        
        print(f"\nStarting experimental study...")
        print(f"Configuration: {replications} replications × {time_limit}s each")
        
        results = []
        all_gaps = []
        all_times = []
        all_lengths = []
        
        for i in range(replications):
            seed = custom_seed + i if custom_seed is not None else i
            result = run_single_experiment(tsp_file, opt_file, delta, seed, time_limit)
            results.append(result)
            
            if result['status'] == 'success' and result['feasible_count'] > 0:
                all_gaps.append(result['best_gap'])
                all_lengths.append(result['best_length'])
            all_times.append(result['solve_time'])
        
        analyze_and_report_results(results, tsp_file, delta, replications)
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    n=input("Press Enter to Exit")
