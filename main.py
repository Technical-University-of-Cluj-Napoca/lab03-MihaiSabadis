from utils import *          # WIDTH, HEIGHT, DEPTH_LIMIT, COLORS
from grid import Grid
from searching_algorithms import *  # bfs, dfs, astar, dls, ucs, greedy, ids, ida_star
import heapq

# --------- Config ---------
CHASE_EXPANSIONS_PER_FRAME = 10  # higher = faster chase, heavier per frame

# --------- Helpers ----------
def update_all_neighbors(grid_obj: Grid) -> None:
    for row in grid_obj.grid:
        for spot in row:
            spot.update_neighbors(grid_obj.grid)

def move_end_one_cell(grid_obj: Grid, end_spot, dcol: int, drow: int):
    """Move END one cell if in-bounds and not a barrier/start."""
    if end_spot is None:
        return None
    nr, nc = end_spot.row + drow, end_spot.col + dcol
    if not (0 <= nr < grid_obj.rows and 0 <= nc < grid_obj.cols):
        return end_spot
    target = grid_obj.grid[nr][nc]
    if target.is_barrier() or target.is_start():
        return end_spot
    end_spot.reset()
    target.make_end()
    return target

# --------- Incremental A* chase engine ----------
class AStarChase:
    """Warm-started A* that keeps frontier/scores as the goal moves."""
    def __init__(self, grid: Grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end

        self.count = 0
        self.open_heap = []      # (f, tie, node)
        self.in_open = set()
        self.came_from = {}
        self.g = {spot: float("inf") for row in grid.grid for spot in row}
        self.g[start] = 0.0

        self._push(start, self._h(start))
        self.caught = False
        self.done = False

    def _h(self, node):
        (x1, y1) = node.get_position()
        (x2, y2) = self.end.get_position()
        return abs(x1 - x2) + abs(y1 - y2)   # Manhattan

    def _push(self, node, f):
        self.count += 1
        heapq.heappush(self.open_heap, (f, self.count, node))
        self.in_open.add(node)

    def update_goal(self, new_end):
        """Call when END moves; we keep frontier & scores."""
        self.end = new_end

    def step(self, draw_cb):
        """Advance A* a few expansions. Call every frame while chasing."""
        if self.caught or self.done:
            return

        for _ in range(CHASE_EXPANSIONS_PER_FRAME):
            # keep QUIT events flowing to outer loop
            for event in pygame.event.get(pygame.QUIT):
                pygame.event.post(event)

            if not self.open_heap:
                self.done = True
                return

            _f, _t, current = heapq.heappop(self.open_heap)
            if current in self.in_open:
                self.in_open.remove(current)

            # goal check against the *current* end position
            if current == self.end:
                cur = current
                while cur in self.came_from:
                    cur = self.came_from[cur]
                    if cur != self.start:
                        cur.make_path()
                        draw_cb()
                self.end.make_end()
                self.start.make_start()
                self.caught = True
                return

            # relax neighbors
            for nb in current.neighbors:
                if nb.is_barrier():
                    continue
                tentative = self.g[current] + 1
                if tentative < self.g[nb]:
                    self.came_from[nb] = current
                    self.g[nb] = tentative
                    f = tentative + self._h(nb)  # heuristic uses the *current* goal
                    self._push(nb, f)
                    if nb != self.end:
                        nb.make_open()

            if current != self.start:
                current.make_closed()

            draw_cb()

# --------- Main ----------
if __name__ == "__main__":
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Path Visualizing Algorithm")

    # Hold-to-move arrows (repeat KEYDOWN while held)
    pygame.key.set_repeat(200, 75)  # (initial delay ms, repeat interval ms)

    ROWS, COLS = 50, 50
    grid = Grid(WIN, ROWS, COLS, WIDTH, HEIGHT)

    start = None
    end = None

    # Chase mode state (only change vs initial: press P to start)
    chasing = False
    engine: AStarChase | None = None

    run = True
    started = False  # used only for one-shot algorithms 1..8

    while run:
        grid.draw()  # base draw (engine also draws during steps)

        # --------------- Events ---------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # When a blocking one-shot algorithm is running, ignore inputs
            if started:
                continue

            # --------- Mouse: left click -> place start/end/barrier ---------
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)
                if 0 <= row < ROWS and 0 <= col < COLS:
                    spot = grid.grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()

            # --------- Mouse: right click -> erase ---------
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)
                if 0 <= row < ROWS and 0 <= col < COLS:
                    spot = grid.grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            # --------- Keys ---------
            if event.type == pygame.KEYDOWN:
                # Clear all
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    chasing = False
                    engine = None
                    grid.reset()
                    continue

                # Start (or restart) chase mode (P) — only new feature
                if event.key == pygame.K_p:
                    if start is not None and end is not None:
                        update_all_neighbors(grid)
                        engine = AStarChase(grid, start, end)
                        chasing = True
                        print("Chase mode: ON (A* incremental). Move END with arrows.")
                    else:
                        print("Place START and END first.")

                # Move END with arrow keys (works always; engine listens if chasing)
                moved_end = False
                if end is not None:
                    if event.key == pygame.K_UP:
                        end = move_end_one_cell(grid, end, -1, 0); moved_end = True
                    elif event.key == pygame.K_DOWN:
                        end = move_end_one_cell(grid, end,  1, 0); moved_end = True
                    elif event.key == pygame.K_LEFT:
                        end = move_end_one_cell(grid, end,  0,-1); moved_end = True
                    elif event.key == pygame.K_RIGHT:
                        end = move_end_one_cell(grid, end,  0, 1); moved_end = True

                if moved_end and chasing and engine is not None:
                    engine.update_goal(end)

                # --- Original single-run algorithms (unchanged behavior) ---
                if event.key in (
                    pygame.K_1, pygame.K_KP1,
                    pygame.K_2, pygame.K_KP2,
                    pygame.K_3, pygame.K_KP3,
                    pygame.K_4, pygame.K_KP4,
                    pygame.K_5, pygame.K_KP5,
                    pygame.K_6, pygame.K_KP6,
                    pygame.K_7, pygame.K_KP7,
                    pygame.K_8, pygame.K_KP8
                ):
                    if start is None or end is None or started:
                        continue

                    update_all_neighbors(grid)
                    started = True
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        bfs(lambda: grid.draw(), grid, start, end)
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        dfs(lambda: grid.draw(), grid, start, end)
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        astar(lambda: grid.draw(), grid, start, end)
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        dls(lambda: grid.draw(), grid, start, end, DEPTH_LIMIT)
                    elif event.key in (pygame.K_5, pygame.K_KP5):
                        ucs(lambda: grid.draw(), grid, start, end)
                    elif event.key in (pygame.K_6, pygame.K_KP6):
                        greedy(lambda: grid.draw(), grid, start, end)
                    elif event.key in (pygame.K_7, pygame.K_KP7):
                        ids(lambda: grid.draw(), grid, start, end, DEPTH_LIMIT)
                    elif event.key in (pygame.K_8, pygame.K_KP8):
                        ida_star(lambda: grid.draw(), grid, start, end)
                    started = False

        # --------------- Per-frame chase stepping ---------------
        if chasing and engine is not None:
            engine.step(lambda: grid.draw())
            if engine.caught:
                print("Caught! Chase mode OFF.")
                chasing = False
                engine = None
            elif engine.done:
                print("Search exhausted; cannot reach target. Chase mode OFF.")
                chasing = False
                engine = None

    pygame.quit()
