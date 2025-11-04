from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot
from math import sqrt,inf

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """

    if start == None or end == None:
        return False

    queue = deque()
    queue.append(start)
    visited = {start}
    came_from = {}
        
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        current = queue.popleft()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()
    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
    """
    Depth-Limited Search (DLS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
        limit (int): The depth limit for the search.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    
    if start ==None or end == None:
        return False

    stack = [(start,0)]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        current, depth = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            
            end.make_end(), start.make_start()
            return True
        if depth < limit:
            for neighbor in current.neighbors:
                if neighbor not in visited and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()
    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Uniform Cost Search (UCS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False
    
    count = 0
    pq = PriorityQueue() #structure: (cost_so_far, tie_breaker, spot)
    pq.put((0, count, start))

    came_from = {} # path reconstruction
    
    # cost from start to spot, initialized to infinity
    cost_so_far = {spot: float("inf") for row in grid.grid for spot in row}
    cost_so_far[start] = 0

    while not pq.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        current_cost, _, current_spot = pq.get()

        if current_cost > cost_so_far[current_spot]:
            continue

        if current_spot == end:
            while current_spot in came_from:
                current_spot = came_from[current_spot]
                if current_spot != start:
                    current_spot.make_path()
                    draw()
            start.make_start()
            end.make_end()
            return True
        
        for neighbor in current_spot.neighbors:
            if neighbor.is_barrier():
                continue

            new_cost = cost_so_far[current_spot] + 1  # assume each step has a cost of 1
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current_spot
                count += 1
                pq.put((new_cost, count, neighbor))

                if neighbor != end:
                    neighbor.make_open()
        if current_spot != start and current_spot != end:
            current_spot.make_closed()

        draw()

    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False

    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        current = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            
            end.make_end(), start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()
    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1,y1 = p1
    x2,y2 = p2

    return abs(x1-x2)+abs(y1-y2)

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1,y1 = p1
    x2,y2 = p2

    return sqrt((x2-x1)**2 + (y2-y1)**2)

def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False
    
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid.grid for spot in row}
    f_score[start] = h_manhattan_distance(start.get_position(), end.get_position())
    open_set = {start} # O(1) check for items in the heap

    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        current  = open_heap.get()[2]
        open_set.remove(current)
        if current == end:
            # reconstruct path
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            tentative_g = g_score[current] + 1  # assume each step has a cost of 1
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h_manhattan_distance(neighbor.get_position(), end.get_position())
                if neighbor not in open_set:
                    count += 1
                    open_heap.put((f_score[neighbor], count, neighbor))
                    open_set.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False  # no path found

#Greedy Best-First Search Algorithm(Dijkstra)
def greedy(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Greedy Best-First Search Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False
    
    count = 0
    pq = PriorityQueue()
    pq.put((0, count, start))
    came_from = {}
    visited = {start} # O(1) check for items in the heap

    while not pq.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        _, _, current = pq.get()

        if current == end:
            # reconstruct path
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                came_from[neighbor] = current
                count += 1
                priority = h_manhattan_distance(neighbor.get_position(), end.get_position())
                pq.put((priority, count, neighbor))
                visited.add(neighbor)
                neighbor.make_open()

                if neighbor != end:
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False  # no path found

def ids(draw: callable, grid: Grid, start: Spot, end: Spot, max_depth: int) -> bool:
    """
    Iterative Deepening Search (IDS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
        max_depth (int): The maximum depth limit for the search.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    for depth in range(max_depth):
        if dls(draw, grid, start, end, depth):
            return True
    return False

def ida_star(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Iterative Deepening A* (IDA*) Algorithm.
    """
    if start is None or end is None:
        return False

    def dfs_ida(current: Spot, g_cost: float, threshold: float) -> tuple[bool, float]:
        # keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False, inf

        f_cost = g_cost + h_manhattan_distance(current.get_position(), end.get_position())

        if f_cost > threshold:
            return False, f_cost

        if current == end:
            return True, f_cost

        min_threshold = float("inf")

        for neighbor in current.neighbors:
            if neighbor.is_barrier() or neighbor in visited:
                continue

            visited.add(neighbor)
            
            came_from[neighbor] = current # record parent BEFORE going deeper

        
            if neighbor != end:
                neighbor.make_open()
            draw()

            found, temp_threshold = dfs_ida(neighbor, g_cost + 1, threshold)

           
            visited.remove(neighbor) # backtrack in visited for this branch

            if found:
                return True, temp_threshold

            if temp_threshold < min_threshold:
                min_threshold = temp_threshold

        return False, min_threshold

    # initial threshold = heuristic(start, goal)
    threshold = h_manhattan_distance(start.get_position(), end.get_position())
    came_from = {}
    visited = {start}

    while True:
        found, temp_threshold = dfs_ida(start, 0, threshold)

        if found:
            # reconstruct path
            current = end
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                    draw()
            start.make_start()
            end.make_end()
            return True

        if temp_threshold == inf or temp_threshold <= threshold:
            return False

        threshold = temp_threshold