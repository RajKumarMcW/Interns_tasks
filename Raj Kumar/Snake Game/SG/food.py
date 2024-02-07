import random

class Food:
    def __init__(self,row,col,snake_position):
        x=random.randint(0,row-1)
        y=random.randint(0,col-1)
        Food.position=[x,y]
        if Food.position in snake_position:
            food = Food(row,col,snake_position)
