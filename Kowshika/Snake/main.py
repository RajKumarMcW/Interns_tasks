from snake import S
from board import Board
from food import Food
import os
import time
import keyboard
os.system("cls")
while(1):

    # direction=input("Enter direction using the keys W,A,S,D")
    choice=int(input(f"""\n\n\n\n
                     Press 1 to Start 
                     Press 2 to Exit
                     \n\n\n\n"""))
    if(choice==1):
        n=12
        b=Board(n)
        f=Food()
        s=S()
        while(1):
            os.system("cls")
            if keyboard.is_pressed(" "):
                print(f"""\n\n\n\n\n\n
                      Paused
                      \n\n\n\n
                      Press Space to Resume...\n
                      """)
                keyboard.wait(" ")
            if keyboard.is_pressed("ESC"):
                print(f"""\n\n\n\n\n
                      Press 'y' to Exit the game and 'n' to cancel
                      \n\n\n\n\n
                      """)
                if(input()=='y'):
                    choice=1
                    break
            if(s.change_direction()):
                print(f"""\n\n\n\n\n
                      Game Over.
                      Your score is {len(s.coordinate)-1}
                      \n\n\n\n\n""")
                time.sleep(1)
                break
            b.display_board(s.coordinate,f.position)
            time.sleep(0.15)
    elif(choice==2):
        break
    else:
        print("Enter valid Input please...")
