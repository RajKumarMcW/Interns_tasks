from collections import deque
import random

class Maze:
    maze_list=[]
    def __init__(self,row):
        assert row>0, "rows should greater than 0"
        self.__row=row
        self.__matrix = self.__generate_matrix()
        print("matrix created")
        self.display_maze([])

        Maze.maze_list.append(self)
    
    def __generate_matrix(self):
        matrix= [['#' for j in range(self.__row)] for i in range(self.__row)]
        # matrix= [[random.choice(['#','X']) for j in range(self.__row)] for i in range(self.__row)]
        return matrix


    def shortest_path(self,start, end): 
        visited = [[False] * self.__row for _ in range(self.__row)]
        queue = deque([(start[0], start[1], 0, [])])  

        while queue:
            x, y, distance, path = queue.popleft()

            # Check if the current position is the destination
            if (x, y) == end:
                return distance, path + [(x, y)]

            # Mark the current position as visited
            visited[x][y] = True

            # Explore neighboring cells
            neighbors = [ (x - 1, y-1), (x - 1, y),(x - 1, y+1),
                        (x, y - 1), (x, y + 1),
                        (x + 1, y-1), (x + 1, y),(x + 1, y+1)]
            for nx, ny in neighbors:
                if 0 <= nx < self.__row and 0 <= ny < self.__row and self.__matrix[nx][ny] == '#' and not visited[nx][ny]:
                    queue.append((nx, ny, distance + 1, path + [(x, y)]))
                    visited[nx][ny] = True

        # If no path is found
        return -1, []
    

    def display_maze(self,path):
        for i in range(self.__row):
            for j in range(self.__row):
                if((i,j) in path):
                    self.__matrix[i][j]='+'
                print(f' {self.__matrix[i][j]}', end='')
                # if(j==self.__row-1):
                #     continue
                # print(" |",end='')
            
            print()
            # if(i==self.__row-1):
            #     continue
            # for j in range(self.__row):
            #     print("----", end='')
            # print()

    def __repr__(self):
        return f"""
        Rows:{self.__row} 
        Matrix:
        {self.__matrix}"""



