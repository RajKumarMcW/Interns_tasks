from SG.board import Board
import os

def print_instruction():
    os.system("cls")
    print(f"""Controls:
          LEFT    : \u2190  
          UP      : \u2191  
          RIGHT   : \u2192  
          DOWN    : \u2193
          OPTION  : ESC
          \n\n\n\n\n
Enter any key to exit...""")

choice='1'
while(1): 
    os.system("cls")
    if(choice not in ['1','2','3','4','5']):
        print("Enter Valid Input..")
    choice=input(f"""\n\n\n\n\n
        Enter 1 - Start Game
        Enter 2 - AI Mode
        Enter 3 - Controls
        Enter 4 - History
        Enter 5 - Exit Game
        \n\n\n\n\n
        """)
    if(choice=='1'):                
        b1=Board(1)
    
    elif(choice=='2'):
        b1=Board(2)

    elif(choice=='3'):
        print_instruction()
        if(input()):
            pass

    elif(choice=='4'):
        os.system("cls")
        with open('score.txt',"r") as f:
            hscore=f.readline()
            hscore=[num for num in hscore.split(',')]
            scr=f.read()
            print(f"""Highest score: {hscore[0]}
Name: {hscore[1]} 
Time:{hscore[2]}\n\n
previous score:
{scr}""")
        if(input()):
            pass


    elif(choice=='5'):
        os.system("cls")

        break
        