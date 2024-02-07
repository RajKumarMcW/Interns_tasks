import os
import time
import keyboard
from snake import Snake
from food import Food
from datetime import datetime


class Board:
    row=7
    col=7
    def __init__(self,choice):        
        self.__matrix =  [[' ' for j in range(self.row)] for i in range(self.col)]
        s1=Snake()
        food=Food(self.row,self.col,s1.coordinates)
        while(1):
            if keyboard.is_pressed("ESC"):
                print(f"""\n Exit :Enter 'y' \n Resume : Enter 'n' \n""")
                if(input()=='y'):
                    os.system("cls")
                    break
            os.system("cls")
            
            # self.writeFile1(s1,food)
            if(s1.change_direction(self.row,self.col,choice)):
                self.game_over(len(s1.coordinates)-1)
                with open('SG\\score.txt',"r") as f:
                    high_score_line=f.readline()
                    high_score=[num for num in high_score_line.split(',')]
                    content=f.read()

                name=input("Enter your name:")
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
                

                with open('SG\\score.txt',"w") as f:
                    if(len(s1.coordinates)-1>int(high_score[0])):
                        print("You reached highest score")
                        f.write(f"{str(len(s1.coordinates)-1)},{name},{formatted_datetime}")
                    else:
                        f.write(f"{high_score[0]},{high_score[1]},{high_score[2]}")
                    f.write(f"{str(len(s1.coordinates)-1)},{name},{formatted_datetime}\n")
                    f.write(f"{content}")

                break

            # self.writeFile2(s1)
            self.display_board(s1.coordinates,food.position)
            time.sleep(0.3)

    def writeFile1(self,s1,food):
        with open("log.csv","a") as f:
            sx,sy=s1.coordinates[0]
            fx,fy=food.position
            up,down,left,right=0,0,0,0
            if(sx>fx):
                up=1
            if(sx<fx):
                down=1
            if(sy>fy):
                left=1
            if(sy<fy):
                right=1
            if s1.direction=='w':
                dir=1
            elif s1.direction=='s':
                dir=2
            elif s1.direction=='a':
                dir=3
            else:
                dir=4
            f.write(f"{sx},{sy},{up},{down},{left},{right},{dir},")

    def writeFile2(self,s1):
        with open("log.csv","a") as f:
                if s1.direction=='w':
                    dir=1
                elif s1.direction=='s':
                    dir=2
                elif s1.direction=='a':
                    dir=3
                else:
                    dir=4
                f.write(f"{dir}\n")

    def game_over(self,score):
        print(f"""\n\n\n\n\n
                Game Over
                Score: {score}
                \n\n\n\n\n
                """)
        time.sleep(2)
        os.system("cls")

    def display_board(self,snake_position,food_position):
        print(f"""Options:Esc""")
        for i in range(-1,self.row+1):
            for j in range(-1,self.col+1):
                #boundary
                if((i==-1 or i==self.row) and j!=self.col and j!=-1 ):
                    print("---",end='')
                elif(j==-1 or j==self.col):
                    print('|',end='')
                
                #snake head
                elif([i,j] == snake_position[0]):
                    print(" 0 ",end='')

                #snake body
                elif([i,j] in snake_position):
                    print(" * ",end='')

                #food
                elif([i,j]==food_position):
                    print(" # ",end='')
                
                else:
                    print(f' {self.__matrix[i][j]} ', end='')
            print()  
        print(f"""Score:{len(snake_position)-1}""")        