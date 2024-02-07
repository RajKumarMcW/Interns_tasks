from food import Food
from board import Board
import keyboard
class S:
    def __init__(self):
        self.size=1
        self.coordinate=[[0,0] for i in range(0,self.size)]
        self.direction='d'
    def change_direction(self):
        if keyboard.is_pressed('LEFT'):
            if self.direction!='d':
                self.direction='a'
        elif keyboard.is_pressed('RIGHT'):
            if self.direction!='a':
                self.direction='d'
        elif keyboard.is_pressed('UP'):
            if self.direction!='s':
                self.direction='w'
        elif keyboard.is_pressed('DOWN'):
            if self.direction!='w':
                self.direction='s'
        x,y=self.coordinate[0]
        if self.direction=='w':
            x-=1
        elif self.direction=='s':
            x+=1
        elif self.direction=='a':
            y-=1
        elif self.direction=='d':
            y+=1
        if x<0 or y<0 or x>Board.row-1 or y>Board.row-1:
            return True
        elif [x,y] in self.coordinate:
            return True
        self.coordinate.insert(0,[x,y])
        if x==Food.position[0] and y==Food.position[1]:
            while(1):
                food=Food()
                if Food.position not in self.coordinate:
                    break
        else:
            del self.coordinate[-1]
