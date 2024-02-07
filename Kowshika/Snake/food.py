import random
from board import Board
class Food:
    def __init__(self):
        x=random.randint(0,Board.row-1)
        y=random.randint(0,Board.row-1)
        Food.position=[x,y]
    