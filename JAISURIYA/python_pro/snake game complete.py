from threading import Thread
import random
import os
from pynput import keyboard
import threading
import time


class Direction:
    def __init__(self,dir):
          self.dir = dir

class Food:
    def __init__(self, n):
        self.n = n
        self.x_food = random.randint(1,n-1)
        self.y_food = random.randint(1,n-1)

class Head:
    def __init__(self,x = 1,y = 1):
        self.head_x = x
        self.head_y = y

class Snake:
    def __init__(self):
        self.pos = [[0,0]]
        self.score = 0


class Move(Head,Food,Snake,Direction):
    def __init__(self,n):
        self.m = n
        Head.__init__(self,0, 0)
        Snake.__init__(self)
        Direction.__init__(self,'r')
        Food.__init__(self,self.m)
        self.last_pos = []
        self.flag = True

    def updatepos(self):
        self.last_pos = self.pos.pop()
        self.pos.insert(0,[self.head_x,self.head_y])

    def invalid(self):
         self.flag = False

    def isvalid(self,x,y):
         if [x,y] not in self.pos:
              return True
         return False

    def moving(self):
        if self.dir == 't':
                if self.head_x-1>=0 and self.isvalid(self.head_x-1,self.head_y):
                    self.head_x-=1
                    self.updatepos()
                else:
                     self.invalid()
        elif self.dir == 'd':
                if self.head_x+1<self.n and self.isvalid(self.head_x+1,self.head_y):
                    self.head_x+=1
                    self.updatepos()
                else:
                    self.invalid()
        elif self.dir == 'r':
                if self.head_y+1<self.n and self.isvalid(self.head_x,self.head_y+1):
                    self.head_y+=1
                    self.updatepos()
                else:
                     self.invalid()
        elif self.dir == 'l':
                if self.head_y-1>=0 and self.isvalid(self.head_x,self.head_y-1):
                    self.head_y-=1
                    self.updatepos()
                else:
                     self.invalid()

        if self.head_x == self.x_food and self.head_y == self.y_food:
            self.score+=1
            self.pos.append(self.last_pos)
            while [self.x_food,self.y_food] in self.pos:
                Food.__init__(self,self.m)
        return self.flag
    
class printGrid(Move):
    
    def __init__(self,n):
        Move.__init__(self,n)

    def print_snake_pos(self):
        for i in range(self.n):
            for j in range(self.n):
                if i == self.head_x and j == self.head_y:
                    print("H ",end = " ")
                elif [i,j] in self.pos:
                    print("0 ",end = " ")
                elif i == self.x_food and j == self.y_food:
                    print("X ",end = " ")
                else:
                    print(". ",end = " ")
            print()


listener_running = True
pause = False

def on_press(key):
    global listener_running
    if key == keyboard.Key.down:
        PrintGrid.dir = 'd'
    elif key == keyboard.Key.up:
        PrintGrid.dir = 't'
    elif key == keyboard.Key.left:
        PrintGrid.dir = 'l'
    elif key == keyboard.Key.right:
        PrintGrid.dir = 'r'
    elif key == keyboard.Key.space:
        global pause
        pause = True
    elif key == keyboard.Key.esc or key == keyboard.KeyCode(char = 'E'):
        listener_running = False
        return False   

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        while listener_running:
            listener.join(1)  

listener_thread = threading.Thread(target=start_listener)
listener_thread.start()

def menu():
    print("MENU")
    print("1  --->  Start Game")
    print("2  --->  Help")
    print("E --->  Exit")
    return input("Enter your option... ")

def game_over():
    print("Game Over")
    print("Returning to Main Menu...")
    print("Please Wait....")
    time.sleep(3)

def help():
    print("Space Bar   ---> Pause the Game")
    print("Up Arrow    ---> Upward Movement")
    print("Down Arrow  ---> Downward Movement")
    print("Left Arrow  ---> Leftside Movement")
    print("Right Arrow ---> Rightside Movement")
    print("H ---> Head of the Snake")
    print("0 ---> Body of the Snake")
    print("Press 0 ---> Go back to main menu")
    return input()

def pause_function():
    print("Press c to continue")
    print("Press 0 -> Main menu")
    return input()

try:
    option = '0'
    while listener_running:
        if(option == '0'):
            option = menu()
        if(option == '1'):
            n = int(input("Enter Board Size : "))
            PrintGrid = printGrid(n)
            n=''
            option = '4'
        elif(option == '4'):
            flag = PrintGrid.moving()
            PrintGrid.print_snake_pos()    
            print()
            time.sleep(0.5)
            while pause:
                os.system('pause')
                pause = False
            if flag == False:
                print(f"Score : {PrintGrid.score}")
                game_over()
                option = '0'
            else:
                os.system('cls')
        elif option == '2':
            option = help()
        else:
            listener_running = False

finally:
    listener_thread.join()
    print("Thanks for Playing...")