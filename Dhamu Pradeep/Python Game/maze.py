class Path:
    def __init__(self,n,start,end):
        self.n = n
        self.paths = [['+' for i in range(n)] for j in range(n)]
        self.start = start
        self.end = end
        self._visited = [[False for i in range(n)] for j in range(n)]
        self._minpath = -1
        self._shortestpaths = [[[] for i in range(n)] for j in range(n)]

    def __isvalid(self,currpos,i,j):
        if currpos[0]+i>=0 and currpos[0]+i<self.n and currpos[1]+j>=0 and currpos[1]+j<self.n and self._visited[currpos[0]+i][currpos[1]+j] == False:
            return True
        return False
        
    def _bfs(self):

        q = []

        if self.__isvalid(start,0,0):
            start.append(0)
            q.append(start)

        self._visited[start[0]][start[1]] = True

        while len(q)>0:

            currpos = q.pop(0)

            rows = [-1,1,0,0]
            cols = [0,0,-1,1]

            if currpos[0] == end[0] and currpos[1] == end[1]:
                self._minpath = currpos[2]
                return
                
            for i in range(4):
                if self.__isvalid(currpos,rows[i],cols[i]):
                    q.append([currpos[0]+rows[i],currpos[1]+cols[i],currpos[2]+1])
                    self._visited[currpos[0]+rows[i]][currpos[1]+cols[i]] = True
                    self._shortestpaths[currpos[0]+rows[i]][currpos[1]+cols[i]] = [currpos[0],currpos[1]]

    

class ShortTestPath(Path):
    def __init__(self,n,start,end):
        super().__init__(n,start,end)
        self.__visited_paths = []

    def __getPaths(self):

        self.paths[self.start[0]][self.start[1]] = '#'

        while(len(self._shortestpaths[self.end[0]][self.end[1]])):
            self.__visited_paths.append([self.end[0],self.end[1]])
            self.paths[self.end[0]][self.end[1]] = '#'
            self.end[0],self.end[1] = self._shortestpaths[self.end[0]][self.end[1]][0],self._shortestpaths[self.end[0]][self.end[1]][1]

        self.__visited_paths.append([self.start[0],self.start[1]])
    
    def __print_matrix(self):
        print("Matrix : ")
        for i in self.paths:
            for j in i:
                print(j,end = " ")
            print()

    def __print_paths(self):
        print("Path from Start to End : ",end = "")
        for i in range(len(self.__visited_paths)-1,1,-1):
            print(self.__visited_paths[i],end = "->")
        print(self.__visited_paths[0])

    def findShortestPath(self):
        super()._bfs()
        self.__getPaths()
        if(self._minpath != -1):
            print(f"Minimum number of steps : {self._minpath}")
            self.__print_paths()
            self.__print_matrix()


n = int(input())

start = input().split()
end = input().split()

start = [int(x) for x in start]
end = [int(x) for x in end]

Obj = ShortTestPath(n,start,end)

Obj.findShortestPath()