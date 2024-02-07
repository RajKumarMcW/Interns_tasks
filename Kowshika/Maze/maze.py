class MazeSolver:
    def __init__(self, maze):
        self.__maze = maze
        self.__rows = len(maze)
        self.__visited = set()
    def __print_maze(self, path=[]):
        for i in range(self.__rows):
            for j in range(self.__rows):
                if (i, j) in path:
                    print("+", end=" ")
                else:
                    print("#" ,end=" ")
            print()
    def find_min_steps(self, start, end):
        queue = [(start, 0, [])]
        while queue:
            current, steps, path = queue.pop(0)
            x, y = current
            if current == end:
                self.__print_maze(path + [end])
                print("Minimum steps:", steps)
                return
            if current not in self.__visited:
                self.__visited.add(current)
                neighbours=[(x-1,y),(x+1,y),(x,y-1),(x,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1),(x+1,y+1)]
                for (a,b) in neighbours:
                    if 0<=a<self.__rows and 0<=b<self.__rows and self.__maze[a][b]==0:
                        queue.append(((a,b),steps+1,path+[(x,y)]))
        print("No valid path found.")



