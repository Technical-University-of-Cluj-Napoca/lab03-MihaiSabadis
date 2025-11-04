from utils import *

class Spot:
    # --- Constructor ---
    def __init__(self, row: int, col: int, width: int, height: int, total_rows: int):
        """
        Initialize a spot in the grid.
        Args: 
            row (int): The row index of the spot.
            col (int): The column index of the spot.
            width (int): The width of the spot.
            height (int): The height of the spot.
            total_rows (int): Keeps track of the total number of rows in the grid (while avoiding global variables).
        """
        # a square has a position in the grid (row, col) and a position in the window (x, y)
        self.row: int = row
        self.col: int = col
        # a square has also a width and a height
        # the coordinates (x, y) are calculated based on the place inside the grid and its size.
        self.width: int = width
        self.height: int = height
        self.x: int = row * width
        self.y: int = col * height
        self.color: tuple = COLORS["WHITE"]  # default color is white
        self.neighbors: list = []
        self.total_rows: int = total_rows

    # ---- Methods to change the state of the spot (i.e., its setters) ----
    def get_position(self) -> tuple[int, int]:
        """
        Gets the (row, col) position of the spot in the grid.
        Returns:
            tuple[int, int]: A tuple containing the row and column indices of the spot in the grid.
        """
        return self.row, self.col

    def is_closed(self) -> bool:
        """
        Checks if the spot is marked as closed (red) i.e. "Have we already looked at you?".
        Returns:
            bool: True if the spot is closed (red), False otherwise.
        """
        return self.color == COLORS['RED']

    def is_open(self) -> bool:
        """
        Checks if the spot is marked as open (green), i.e. "Are you free to pass?".
        Returns:
            bool: True if the spot is marked as open (green), False otherwise.
        """
        return self.color == COLORS['GREEN']

    def is_barrier(self) -> bool:
        """
        Checks if the spot is marked as a barrier (black) i.e. "You cannot go through me!".
        Returns:
            bool: True if the spot is a barrier (black), False otherwise.
        """
        return self.color == COLORS['BLACK']

    def is_start(self) -> bool:
        """
        Checks if the spot is marked as the start node (orange).
        Returns:
            bool: True if the spot is the start node (orange), False otherwise.
        """
        return self.color == COLORS['ORANGE']

    def is_end(self) -> bool:
        """
        Checks if the spot is marked as the end node (turquoise).
        Returns:
            bool: True if the spot is the end node (turquoise), False otherwise.
        """
        return self.color == COLORS['TURQUOISE']

    # ---- Methods to change the state of the spot (i.e., its setters) ----
    def reset(self) -> None:
        """
        Change the color of the spot back to white (unvisited).
        Returns:
            None
        """
        self.color = COLORS['WHITE']

    def make_closed(self) -> None:
        """
        Mark the spot as closed (red).
        Returns:
            None
        """
        self.color = COLORS['RED']

    def make_open(self) -> None:
        """
        Mark the spot as open (green).
        Returns:
            None
        """
        self.color = COLORS['GREEN']

    def make_barrier(self) -> None:
        """
        Mark the spot as a barrier (black).
        Returns:
            None
        """
        self.color = COLORS['BLACK']

    def make_start(self) -> None:
        """
        Mark the spot as the start node (orange).
        Returns:
            None
        """
        self.color = COLORS['ORANGE']

    def make_end(self) -> None:
        """
        Mark the spot as the end node (yellow).
        Returns:
            None
        """
        self.color = COLORS['YELLOW']

    def make_path(self) -> None:
        """
        Mark the spot as part of the path (purple).
        Returns:
            None
        """
        self.color = COLORS['PURPLE']

    # --- Operators ---
    # "Spot" type is not yet defined because the class will be defined at runtime and will exist only after it is closed (the whole class).
    # So we use quotes to tell the type checker that this is a string, containing the name of a type that will exist later.
    def __lt__(self, other: "Spot") -> bool:
        """
        Less-than operator for comparing two spots. The other spot is always "greater" than this one.
        This is used to avoid errors in data structures that require comparison, like PriorityQueue.
        """
        return False
    
    # --- Other Methods ---
    def draw(self, win: pygame.Surface) -> None:
        """
        Draw the spot on the given Pygame surface (window).
        Args:
            win (pygame.Surface): The Pygame surface (window) where the spot will be drawn.
        """
        # draw a rectangle at (x, y) with size (width, width) and color self.color
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid: list[list["Spot"]]) -> None:
        """
        Update the list of neighbor spots that are not barriers.
        Args:
            grid (list[list[Spot]]): The 2D list (matrix) representing the grid of Spot objects.
        Returns:
            None
        """
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])