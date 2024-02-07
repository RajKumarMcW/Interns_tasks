from maze import MazeSolver
n = int(input("Enter the size of the maze (n): "))
maze = []
print("Enter the maze elements (0 for clear path, 1 for wall):")
for i in range(n):
    row = list(map(int, input().split()))
    maze.append(row)
start = tuple(map(int, input("Enter the starting point (x y): ").split()))
end = tuple(map(int, input("Enter the ending point (x y): ").split()))
solver = MazeSolver(maze)
solver.find_min_steps(start, end)