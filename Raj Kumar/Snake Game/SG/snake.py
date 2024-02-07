# from board import Board
from SG.food import Food
import pandas as pd
import joblib
import keyboard

clf =  joblib.load('trained.joblib')


def preprocess_input(sx,sy,up,down,left,right,dir):
    # Convert input to DataFrame
    input_data = pd.DataFrame({
        'snake_head_pos_x': [sx],
        'snake_head_pos_y': [sy],
        'food_pos_up': [up],
        'food_pos_down': [down],
        'food_pos_left': [left],
        'food_pos_right': [right],
        'direction':[dir]
    })
    return input_data

def predict_direction(sx,sy,up,down,left,right,dir):
    # Preprocess input data
    input_data = preprocess_input(sx,sy,up,down,left,right,dir)

    # Predict direction
    prediction = clf.predict(input_data)[0]

    print(prediction)
    return prediction

class Snake:

    def __init__(self):
        self.size=1
        self.coordinates=[[0,0] for i in range(0,self.size)]
        self.direction=4

    def change_direction(self,row,col,choice):
        sx,sy=self.coordinates[0]
        fx,fy=Food.position
        up,down,left,right=0,0,0,0
        if(sx>fx):
            up=1
        if(sx<fx):
            down=1
        if(sy>fy):
            left=1
        if(sy<fy):
            right=1
        
        if choice==2:
            self.direction = predict_direction(sx,sy,up,down,left,right,self.direction)
        else:
            self.check()
        
        
        x,y=self.coordinates[0]

        if self.direction==1:
            x-=1
        elif self.direction==2: 
            x+=1
        elif self.direction==3: 
            y-=1
        elif self.direction==4: 
            y+=1

        if x<0 or y<0 or x>=row or y>=col:
            return True
        elif [x,y] in self.coordinates:
            return True
        
        self.coordinates.insert(0,[x,y])

            
        if x==Food.position[0] and y==Food.position[1]:
            food=Food(row,col,self.coordinates)

        else:
            del self.coordinates[-1]

    def check(self):
        if keyboard.is_pressed('RIGHT'):
            if self.direction!=3:
                self.direction=4
        elif keyboard.is_pressed('LEFT'):
            if self.direction!=4:
                self.direction=3
        elif keyboard.is_pressed('UP'):
            if self.direction!=2:
                self.direction=1
        elif keyboard.is_pressed('DOWN'):
            if self.direction!=1:
                self.direction=2