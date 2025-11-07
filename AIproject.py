import heapq
import math
from collections import deque
import time
import matplotlib.pyplot as plt
import networkx as nx

class AmbulanceRoutingSystem:
    def __init__(self):
        self.graph = {}
        self.hospitals = {}
        self.accidents = {}
        self.blocked_roads = set()
        self.priority_zones = {}
        
    def add_location(self, location, connections):
        """Add a location with its connections to other locations"""
        self.graph[location] = connections
        
    def add_hospital(self, name, location):
        """Add a hospital at a specific location"""
        self.hospitals[name] = location
        
    def add_accident(self, id, location):
        """Add an accident at a specific location"""
        self.accidents[id] = location
        
    def block_road(self, location1, location2):
        """Block a road between two locations"""
        self.blocked_roads.add((location1, location2))
        self.blocked_roads.add((location2, location1))
        
    def add_priority_zone(self, zone, multiplier):
        """Add a priority zone with a travel time multiplier"""
        self.priority_zones[zone] = multiplier
        
    def bfs_shortest_path(self, start, goal):
        """Find shortest path using BFS (uniform cost)"""
        if start == goal:
            return [start], 0
            
        queue = deque([(start, [start])])
        visited = set([start])
        nodes_expanded = 0
        
        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1
            
            for neighbor, cost in self.graph.get(current, {}).items():
                # Check if road is blocked
                if (current, neighbor) in self.blocked_roads:
                    continue
                    
                if neighbor == goal:
                    return path + [neighbor], len(path), nodes_expanded
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return None, float('inf'), nodes_expanded
    
    def a_star_search(self, start, goal, heuristic):
        """Find optimal path using A* search"""
        if start == goal:
            return [start], 0, 0
            
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start]))
        
        g_costs = {start: 0}
        visited = set()
        nodes_expanded = 0
        
        while open_set:
            _, g_cost, current, path = heapq.heappop(open_set)
            nodes_expanded += 1
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                return path, g_cost, nodes_expanded
                
            for neighbor, cost in self.graph.get(current, {}).items():
                # Check if road is blocked
                if (current, neighbor) in self.blocked_roads:
                    continue
                    
                # Apply priority zone multiplier if applicable
                zone_multiplier = self.priority_zones.get(neighbor, 1.0)
                new_cost = g_cost + cost * zone_multiplier
                
                if neighbor not in g_costs or new_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_cost
                    h_cost = heuristic(neighbor, goal)
                    f_cost = new_cost + h_cost
                    heapq.heappush(open_set, (f_cost, new_cost, neighbor, path + [neighbor]))
                    
        return None, float('inf'), nodes_expanded
    
    def best_first_search(self, start, goal, heuristic):
        """Find path using Best-First Search (greedy)"""
        if start == goal:
            return [start], 0, 0
            
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), start, [start]))
        
        visited = set()
        nodes_expanded = 0
        total_cost = 0
        
        while open_set:
            h_cost, current, path = heapq.heappop(open_set)
            nodes_expanded += 1
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                # Calculate actual cost of path
                for i in range(len(path) - 1):
                    total_cost += self.graph[path[i]][path[i+1]]
                return path, total_cost, nodes_expanded
                
            for neighbor, cost in self.graph.get(current, {}).items():
                # Check if road is blocked
                if (current, neighbor) in self.blocked_roads:
                    continue
                    
                if neighbor not in visited:
                    h_cost = heuristic(neighbor, goal)
                    heapq.heappush(open_set, (h_cost, neighbor, path + [neighbor]))
                    
        return None, float('inf'), nodes_expanded
    
    def euclidean_heuristic(self, a, b):
        """Euclidean distance heuristic for A*"""
        if hasattr(a, 'x') and hasattr(b, 'x'):
            return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        return 0
    
    def manhattan_heuristic(self, a, b):
        """Manhattan distance heuristic for grid-like maps"""
        if hasattr(a, 'x') and hasattr(b, 'x'):
            return abs(a.x - b.x) + abs(a.y - b.y)
        return 0
    
    def find_nearest_hospital(self, accident_location, algorithm='a_star'):
        """Find the nearest hospital for a given accident location"""
        best_path = None
        best_cost = float('inf')
        best_hospital = None
        nodes_expanded = 0
        
        heuristic = self.euclidean_heuristic
        
        for hospital_name, hospital_location in self.hospitals.items():
            if algorithm == 'bfs':
                path, cost, expanded = self.bfs_shortest_path(accident_location, hospital_location)
            elif algorithm == 'best_first':
                path, cost, expanded = self.best_first_search(accident_location, hospital_location, heuristic)
            else:  # a_star
                path, cost, expanded = self.a_star_search(accident_location, hospital_location, heuristic)
                
            nodes_expanded += expanded
            
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_hospital = hospital_name
                
        return best_hospital, best_path, best_cost, nodes_expanded
    
    def visualize_network(self, path=None):
        """Visualize the road network with hospitals, accidents, and optional path"""
        G = nx.Graph()
        
        # Add nodes and edges
        for location, connections in self.graph.items():
            for neighbor, cost in connections.items():
                G.add_edge(location, neighbor, weight=cost)
        
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(12, 8))
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(G, pos)
        
        # Draw edges with different styles for blocked roads
        regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in self.blocked_roads]
        blocked_edges = [(u, v) for u, v in G.edges() if (u, v) in self.blocked_roads]
        
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, style='solid')
        nx.draw_networkx_edges(G, pos, edgelist=blocked_edges, style='dashed', edge_color='red')
        
        # Highlight hospitals
        hospital_nodes = list(self.hospitals.values())
        nx.draw_networkx_nodes(G, pos, nodelist=hospital_nodes, node_color='green', node_size=700)
        
        # Highlight accidents
        accident_nodes = list(self.accidents.values())
        nx.draw_networkx_nodes(G, pos, nodelist=accident_nodes, node_color='red', node_size=700)
        
        # Highlight path if provided
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=3)
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='blue', node_size=700)
        
        # Add legend
        plt.plot([], [], 'o', color='green', label='Hospitals')
        plt.plot([], [], 'o', color='red', label='Accidents')
        plt.plot([], [], 'o', color='blue', label='Optimal Path')
        plt.plot([], [], '--', color='red', label='Blocked Roads')
        plt.legend()
        
        plt.title("Ambulance Routing Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def demonstrate_system():
    # Create the routing system
    system = AmbulanceRoutingSystem()
    
    # Define locations and connections (simplified city map)
    locations = {
        'A': {'B': 5, 'C': 10},
        'B': {'A': 5, 'C': 3, 'D': 7},
        'C': {'A': 10, 'B': 3, 'D': 4, 'E': 8},
        'D': {'B': 7, 'C': 4, 'E': 2, 'F': 6},
        'E': {'C': 8, 'D': 2, 'F': 5, 'G': 9},
        'F': {'D': 6, 'E': 5, 'G': 4},
        'G': {'E': 9, 'F': 4, 'H': 7},
        'H': {'G': 7, 'I': 6},
        'I': {'H': 6, 'J': 8},
        'J': {'I': 8}
    }
    
    # Add locations to the system
    for location, connections in locations.items():
        system.add_location(location, connections)
    
    # Add hospitals
    system.add_hospital("General Hospital", "D")
    system.add_hospital("City Medical Center", "G")
    system.add_hospital("Community Clinic", "J")
    
    # Add accidents
    system.add_accident("Accident_1", "A")
    system.add_accident("Accident_2", "F")
    
    # Add some blocked roads
    system.block_road("C", "D")
    system.block_road("E", "F")
    
    # Add priority zones (travel time multiplier)
    system.add_priority_zone("B", 0.8)  # 20% faster in priority zone
    system.add_priority_zone("H", 1.2)  # 20% slower in construction zone
    
    # Test different algorithms
    algorithms = ['bfs', 'best_first', 'a_star']
    algorithm_names = ['BFS', 'Best-First Search', 'A* Search']
    
    print("AMBULANCE ROUTING SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    for accident_id, accident_loc in system.accidents.items():
        print(f"\n{accident_id} at Location {accident_loc}:")
        print("-" * 30)
        
        results = []
        for i, algorithm in enumerate(algorithms):
            start_time = time.time()
            hospital, path, cost, nodes_expanded = system.find_nearest_hospital(accident_loc, algorithm)
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results.append({
                'algorithm': algorithm_names[i],
                'hospital': hospital,
                'path': path,
                'cost': cost,
                'nodes_expanded': nodes_expanded,
                'time': execution_time
            })
            
            print(f"{algorithm_names[i]}:")
            print(f"  Nearest Hospital: {hospital}")
            print(f"  Path: {' -> '.join(path)}")
            print(f"  Cost: {cost}")
            print(f"  Nodes Expanded: {nodes_expanded}")
            print(f"  Execution Time: {execution_time:.2f} ms")
            print()
        
        # Visualize the best path (using A* as it's typically most efficient)
        print(f"Visualizing optimal path for {accident_id}...")
        system.visualize_network(results[2]['path'])  # A* result
        
        # Compare algorithm performance
        print("\nAlgorithm Comparison:")
        print("Algorithm           | Cost | Nodes Expanded | Time (ms)")
        print("-" * 55)
        for result in results:
            print(f"{result['algorithm']:18} | {result['cost']:4} | {result['nodes_expanded']:13} | {result['time']:8.2f}")

if __name__ == "__main__":
    demonstrate_system()