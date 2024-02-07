class Board:
    def __init__(self,row):
        assert row>0,"row should be greater than 0"
        self.__row=row
        Board.row=row
        self.__matrix=[[' ' for j in range(self.__row)] for i in range(self.__row)]
    def display_board(self,snake_position,food_position):
        print(f"""Score:{len(snake_position)-1} Pause: Space Exit:Esc""")
        for i in range(-1,self.__row+1):
            for j in range(-1,self.__row+1):
                if((i==-1 or i==self.__row) and j!=self.__row and j!=-1):
                    print('---',end='')
                elif(j==-1 or j==self.__row):
                    print('|',end='')
                elif([i,j] == snake_position[0]):
                    print(' 0 ',end='')
                elif([i,j] in snake_position):
                    print(' * ',end='')
                elif([i,j]==food_position):
                    print(' # ',end='')
                else:
                    print(f' {self.__matrix[i][j]} ',end='')
            print()
        print(f"""LEFT: \u2190 UP: \u2191 RIGHT: \u2192 DOWN:\u2193""")