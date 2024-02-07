from maze import Maze
flag=1

while(flag):
    n= int(input("Enter number of row:"))
    m_obj= Maze(n)
    
    start_point = tuple(map(int, input("Enter start point (row column): ").split()))
    end_point = tuple(map(int, input("Enter end point (row column): ").split()))

    result,path=m_obj.shortest_path(start_point,end_point)

    if result != -1:
        m_obj.display_maze(path)
        print(f"The shortest path from {start_point} to {end_point} is {result} steps.")
        print(path)
    else:
        print(f"No path found from {start_point} to {end_point}.")
    
    flag=int(input("Press 1 to continue..."))



print(Maze.maze_list)