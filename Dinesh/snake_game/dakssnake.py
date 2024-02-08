# import os
# import random
# import keyboard
# import time
# import json

# class SnakeGame:
#     def __init__(self):
#         self.CELLS = []
#         self.snake_body = []
#         self.DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = ()
#         self.speed = 0.5  # Default speed
#         self.game_running = True

#     def print_field(self):
#         os.system('clear')
#         for row in range(self.FIELD_HEIGHT):
#             for col in range(self.FIELD_WIDTH):
#                 current_cell = (row, col)
#                 if current_cell in self.snake_body:
#                     if current_cell == self.snake_body[0]:  # Check if it's the head
#                         print('X', end='')
#                     else:
#                         print('X', end='')
#                 elif current_cell == self.apple_pos:
#                     print('a', end='')
#                 elif col in (0, self.FIELD_WIDTH - 1) or row in (0, self.FIELD_HEIGHT - 1):
#                     print('#', end='')
#                 else:
#                     print(' ', end='')

#             print()  # Move to the next line after printing each row

#     def place_apple(self):
#         col = random.randint(1, self.FIELD_WIDTH - 2)
#         row = random.randint(1, self.FIELD_HEIGHT - 2)
#         while (col, row) in self.snake_body:
#             col = random.randint(1, self.FIELD_WIDTH - 2)
#             row = random.randint(1, self.FIELD_HEIGHT - 2)
#         return (row, col)

#     def start_game(self):
#         while True:
#             print("Choose an option:")
#             print("1. New Game")
#             print("2. Resume Game")
#             print("3. Quit")
#             print("4. Help")
#             print("5. About")
#             print("6. Settings")
#             try:
#                 option_choice = int(input("Enter the number corresponding to your choice: "))
#                 if option_choice == 1:
#                     self.game_running = True
#                     while self.game_running:
#                         self.setup_game()
#                         self.set_difficulty()
#                         self.play_game()
#                     if not self.play_again():
#                         print("Thanks for playing. Goodbye!")
#                         break
#                 elif option_choice == 2:
#                     saved_game_file = 'saved_game.json'
#                     if os.path.exists(saved_game_file):
#                         self.load_game(saved_game_file)
#                         self.game_running = True
#                         self.play_game()
#                         if not self.play_again():
#                             print("Thanks for playing. Goodbye!")
#                             break
#                     else:
#                         print("No saved game found. Please start a new game.")
#                 elif option_choice == 3:
#                     print("Thanks for playing. Goodbye!")
#                     break
#                 elif option_choice == 4:
#                     self.display_help()
#                 elif option_choice == 5:
#                     self.display_about()
#                 elif option_choice == 6:
#                     pass
#                 else:
#                     print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")


#     def display_help(self):
#         print("Help:")
#         print("Use the following keys during the game:")
#         print("'w' - Move up")
#         print("'a' - Move left")
#         print("'s' - Move down")
#         print("'d' - Move right")
#         print("'q' - Quit the game")

#     def display_about(self):
#         print("About:")
#         print("This game was developed by Dinesh and Dhatchayani.")
#         print("Snake Game Version 1.0")
#         print("To understand the instructions of the game press 3.")

#     def setup_game(self):
#         while True:
#             print("Choose snake starting position:")
#             print("1. (0, 0)")
#             print("2. Random position")
#             try:
#                 position_choice = int(input("Enter the number corresponding to your choice: "))
#                 if position_choice in [1, 2]:
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1 or 2.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#         while True:
#             print("Enter the width of the field")
#             try:
#                 self.FIELD_WIDTH = int(input())
#                 print("Enter the height of the field")
#                 try:
#                     self.FIELD_HEIGHT = int(input())
#                     if 7 <= self.FIELD_WIDTH <= 32 and 7 <= self.FIELD_HEIGHT <= 20:
#                         break
#                     else:
#                         print("Area dimensions should be between 7X7 and 32X20. Please try again.")
#                 except ValueError:
#                     print("Please enter a valid integer for the height.")
#             except ValueError:
#                 print("Please enter a valid integer for the width.")

#         self.CELLS = [(row, col) for row in range(self.FIELD_HEIGHT) for col in range(self.FIELD_WIDTH)]

#         # snake starting position
#         if position_choice == 1:
#             self.snake_body = [(1, 1)]
#         elif position_choice == 2:
#             valid_start_positions = [(row, col) for row in range(2, self.FIELD_HEIGHT - 2) for col in range(2, self.FIELD_WIDTH - 2)]
#             self.snake_body = [random.choice(valid_start_positions)]

#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = self.place_apple()

#         # Print the initial state
#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def save_game(self, file_path):
#         game_state = {
#             'snake_body': self.snake_body,
#             'direction': self.direction,
#             'eaten': self.eaten,
#             'apple_pos': self.apple_pos,
#             'speed': self.speed,
#             'game_running': self.game_running,
#             'FIELD_WIDTH': self.FIELD_WIDTH,
#             'FIELD_HEIGHT': self.FIELD_HEIGHT
#         }

#         with open(file_path, 'w') as file:
#             json.dump(game_state, file)

#     def load_game(self, file_path):
#         with open(file_path, 'r') as file:
#             game_state = json.load(file)

#         self.snake_body = game_state['snake_body']
#         self.direction = game_state['direction']
#         self.eaten = game_state['eaten']
#         self.speed = game_state['speed']
#         self.game_running = game_state['game_running']
#         self.FIELD_WIDTH = game_state['FIELD_WIDTH']
#         self.FIELD_HEIGHT = game_state['FIELD_HEIGHT']

#         # Place the apple using the current snake's body
#         self.apple_pos = self.place_apple()

#         os.system('clear')
#         print("\nGame loaded:")
#         self.print_field()


#     def set_difficulty(self):
#         print("Choose difficulty level:")
#         print("1. Easy")
#         print("2. Intermediate")
#         print("3. Hard")

#         while True:
#             try:
#                 choice = int(input("Enter the number corresponding to your choice: "))
#                 if choice in [1, 2, 3]:
#                     if choice == 1:
#                         self.speed = 0.5  # Set speed for Easy
#                     elif choice == 2:
#                         self.speed = 0.3  # Set speed for Intermediate
#                     elif choice == 3:
#                         self.speed = 0.1  # Set speed for Hard
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1, 2, or 3.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def play_game(self):
#         total_apples_eaten = 0

#         def on_key_press(e):
#             nonlocal total_apples_eaten
#             if e.event_type == keyboard.KEY_DOWN:
#                 if e.name == 'w':
#                     self.direction = self.DIRECTIONS['up']
#                 elif e.name == 'a':
#                     self.direction = self.DIRECTIONS['left']
#                 elif e.name == 's':
#                     self.direction = self.DIRECTIONS['down']
#                 elif e.name == 'd':
#                     self.direction = self.DIRECTIONS['right']
#                 elif e.name == 'q':
#                     self.game_running = False
#                 elif e.name == 'p':  # Save key
#                     self.save_game('saved_game.json')
#                 elif e.name == 'l':  # Load key
#                     self.load_game('saved_game.json')

#         keyboard.hook(on_key_press)

#         while self.game_running:
#             prev_snake_body = self.snake_body.copy()
#             new_head = (
#                 self.snake_body[0][0] + self.direction[0],
#                 self.snake_body[0][1] + self.direction[1]
#             )
#             self.snake_body.insert(0, new_head)
#             if not self.eaten:
#                 self.snake_body.pop(-1)
#             self.eaten = False

#             # check if snake eats the apple
#             if self.snake_body[0] == self.apple_pos:
#                 self.apple_pos = self.place_apple()
#                 self.eaten = True
#                 total_apples_eaten += 1

#             # check if the snake bites itself
#             if self.snake_body[0] in self.snake_body[1:]:
#                 os.system('clear')
#                 print("\nSnake bites itself. Game over.")
#                 break

#             # check if the snake crosses the boundary
#             if (
#                 self.snake_body[0][0] in (0, self.FIELD_HEIGHT - 1)
#                 or self.snake_body[0][1] in (0, self.FIELD_WIDTH - 1)
#             ):
#                 os.system('clear')
#                 print("\nYou crossed the boundary. Game over.")
#                 break

#             self.print_field()
#             time.sleep(self.speed)  # Use the selected speed for the delay

#         keyboard.unhook_all()
#         print("Total Apples Eaten:", total_apples_eaten)

#     def play_again(self):
#         while True:
#             play_again = input("Do you want to play again? (yes/no): ")
#             if play_again.lower() == 'yes':
#                 self.eaten = False
#                 return True
#             elif play_again.lower() == 'no':
#                 return False
#             else:
#                 print("Invalid input. Please enter 'yes' or 'no.'")



# snake_game = SnakeGame()
# snake_game.start_game()

############################################################################################################################ arrow keys 

# import os
# import random
# import keyboard
# import time
# import json
# from colorama import Fore, init

# class SnakeGame:
#     def __init__(self):
#         self.CELLS = []
#         self.snake_body = []
#         self.DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = ()
#         self.speed = 0.5  # Default speed
#         self.game_running = True
#         self.position_choice = 1  # Default starting position

#     def print_field(self):
#         os.system('clear')
#         for row in range(self.FIELD_HEIGHT):
#             for col in range(self.FIELD_WIDTH):
#                 current_cell = (row, col)
#                 if current_cell in self.snake_body:
#                     if current_cell == self.snake_body[0]:  # Check if it's the head
#                         print(Fore.MAGENTA + 'X', end='')
#                     else:
#                         print(Fore.YELLOW + 'X', end='')
#                 elif current_cell == self.apple_pos:
#                     print(Fore.RED + '*', end='')
#                 elif col in (0, self.FIELD_WIDTH - 1) or row in (0, self.FIELD_HEIGHT - 1):
#                     print(Fore.CYAN+'#', end='')  # Boundaries without color
#                 else:
#                     print(' ', end='')

#             print()  # Move to the next line after printing each row



#     def place_apple(self):
#         col = random.randint(1, self.FIELD_WIDTH - 2)
#         row = random.randint(1, self.FIELD_HEIGHT - 2)
#         while (col, row) in self.snake_body:
#             col = random.randint(1, self.FIELD_WIDTH - 2)
#             row = random.randint(1, self.FIELD_HEIGHT - 2)
#         return (row, col)

#     def start_game(self):
#         while True:
#             print("Choose an option:")
#             print("1. New Game")
#             print("2. Resume Game")
#             print("3. Quit")
#             print("4. Help")
#             print("5. About")
#             print("6. Settings")
#             try:
#                 option_choice = int(input("Enter the number corresponding to your choice: "))
#                 if option_choice == 1:
#                     self.game_running = True
#                     while self.game_running:
#                         self.setup_game()
#                         self.set_difficulty()
#                         self.play_game()
#                     if not self.play_again():
#                         print("Thanks for playing. Goodbye!")
#                         break
#                 elif option_choice == 2:
#                     saved_game_file = 'saved_game.json'
#                     if os.path.exists(saved_game_file):
#                         self.load_game(saved_game_file)
#                         self.game_running = True
#                         self.play_game()
#                         if not self.play_again():
#                             print("Thanks for playing. Goodbye!")
#                             break
#                     else:
#                         print("No saved game found. Please start a new game.")
#                 elif option_choice == 3:
#                     print("Thanks for playing. Goodbye!")
#                     break
#                 elif option_choice == 4:
#                     self.display_help()
#                 elif option_choice == 5:
#                     self.display_about()
#                 elif option_choice == 6:
#                     self.configure_settings()
#                 else:
#                     print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def display_help(self):
#         print("Help:")
#         print("Use the following keys during the game:")
#         print("'w' - Move up")
#         print("'a' - Move left")
#         print("'s' - Move down")
#         print("'d' - Move right")
#         print("'q' - Quit the game")

#     def display_about(self):
#         print("About:")
#         print("This game was developed by Dinesh and Dhatchayani.")
#         print("Snake Game Version 1.0")
#         print("To understand the instructions of the game press 3.")

#     def configure_settings(self):
#         while True:
#             print("Choose snake starting position:")
#             print("1. (0, 0)")
#             print("2. Random position")
#             try:
#                 self.position_choice = int(input("Enter the number corresponding to your choice: "))
#                 if self.position_choice in [1, 2]:
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1 or 2.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def setup_game(self):
#         while True:
#             print("Enter the width of the field")
#             try:
#                 self.FIELD_WIDTH = int(input())
#                 print("Enter the height of the field")
#                 try:
#                     self.FIELD_HEIGHT = int(input())
#                     if 7 <= self.FIELD_WIDTH <= 32 and 7 <= self.FIELD_HEIGHT <= 20:
#                         break
#                     else:
#                         print("Area dimensions should be between 7X7 and 32X20. Please try again.")
#                 except ValueError:
#                     print("Please enter a valid integer for the height.")
#             except ValueError:
#                 print("Please enter a valid integer for the width.")

#         self.CELLS = [(row, col) for row in range(self.FIELD_HEIGHT) for col in range(self.FIELD_WIDTH)]

#         # snake starting position
#         if self.position_choice == 1:
#             self.snake_body = [(1, 1)]
#         elif self.position_choice == 2:
#             valid_start_positions = [(row, col) for row in range(2, self.FIELD_HEIGHT - 2) for col in range(2, self.FIELD_WIDTH - 2)]
#             self.snake_body = [random.choice(valid_start_positions)]

#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = self.place_apple()

#         # Print the initial state
#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def save_game(self, file_path):
#         game_state = {
#             'snake_body': self.snake_body,
#             'direction': self.direction,
#             'eaten': self.eaten,
#             'apple_pos': self.apple_pos,
#             'speed': self.speed,
#             'game_running': self.game_running,
#             'FIELD_WIDTH': self.FIELD_WIDTH,
#             'FIELD_HEIGHT': self.FIELD_HEIGHT,
#             'position_choice': self.position_choice
#         }

#         with open(file_path, 'w') as file:
#             json.dump(game_state, file)

#     def load_game(self, file_path):
#         with open(file_path, 'r') as file:
#             game_state = json.load(file)

#         self.snake_body = game_state['snake_body']
#         self.direction = game_state['direction']
#         self.eaten = game_state['eaten']
#         self.speed = game_state['speed']
#         self.game_running = game_state['game_running']
#         self.FIELD_WIDTH = game_state['FIELD_WIDTH']
#         self.FIELD_HEIGHT = game_state['FIELD_HEIGHT']
#         self.position_choice = game_state['position_choice']

#         # Place the apple using the current snake's body
#         self.apple_pos = self.place_apple()

#         os.system('clear')
#         print("\nGame loaded:")
#         self.print_field()

#     def set_difficulty(self):
#         print("Choose difficulty level:")
#         print("1. Easy")
#         print("2. Intermediate")
#         print("3. Hard")

#         while True:
#             try:
#                 choice = int(input("Enter the number corresponding to your choice: "))
#                 if choice in [1, 2, 3]:
#                     if choice == 1:
#                         self.speed = 0.5  # Set speed for Easy
#                     elif choice == 2:
#                         self.speed = 0.3  # Set speed for Intermediate
#                     elif choice == 3:
#                         self.speed = 0.1  # Set speed for Hard
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1, 2, or 3.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def play_game(self):
#         total_apples_eaten = 0

#         def on_key_press(e):
#             nonlocal total_apples_eaten
#             if e.event_type == keyboard.KEY_DOWN:
#                 if e.name in ['w', 'up']:
#                     self.direction = self.DIRECTIONS['up']
#                 elif e.name in ['a', 'left']:
#                     self.direction = self.DIRECTIONS['left']
#                 elif e.name in ['s', 'down']:
#                     self.direction = self.DIRECTIONS['down']
#                 elif e.name in ['d', 'right']:
#                     self.direction = self.DIRECTIONS['right']
#                 elif e.name == 'q':
#                     self.game_running = False
#                 elif e.name == 'p':  # Save key
#                     self.save_game('saved_game.json')
#                 elif e.name == 'l':  # Load key
#                     self.load_game('saved_game.json')

#         keyboard.hook(on_key_press)

#         while self.game_running:
#             prev_snake_body = self.snake_body.copy()
#             new_head = (
#                 self.snake_body[0][0] + self.direction[0],
#                 self.snake_body[0][1] + self.direction[1]
#             )
#             self.snake_body.insert(0, new_head)
#             if not self.eaten:
#                 self.snake_body.pop(-1)
#             self.eaten = False

#             # check if snake eats the apple
#             if self.snake_body[0] == self.apple_pos:
#                 self.apple_pos = self.place_apple()
#                 self.eaten = True
#                 total_apples_eaten += 1

#             # check if the snake bites itself
#             if self.snake_body[0] in self.snake_body[1:]:
#                 os.system('clear')
#                 print("\nSnake bites itself. Game over.")
#                 self.game_running = False
#                 # break

#             # check if the snakes moves in reverse direction
#             if len(prev_snake_body) > 1 and (
#                     new_head[0] == prev_snake_body[1][0] and new_head[1] == prev_snake_body[1][1]
#                 ):
#                 os.system('clear')
#                 print("\nReverse motion. Game over.")
#                 self.game_running = False
#                 break
            
#             # check if the snake crosses the boundary
#             if (
#                 self.snake_body[0][0] in (0, self.FIELD_HEIGHT - 1)
#                 or self.snake_body[0][1] in (0, self.FIELD_WIDTH - 1)
#             ):
#                 os.system('clear')
#                 print("\nYou crossed the boundary. Game over.")
#                 self.game_running = False
#                 break

#             self.print_field()
#             time.sleep(self.speed)  # Use the selected speed for the delay

#         keyboard.unhook_all()
#         print("Total Apples Eaten:", total_apples_eaten)

#     def play_again(self):
#         while True:
#             play_again = input("Do you want to play again? (yes/no): ")
#             if play_again.lower() == 'yes':
#                 self.eaten = False
#                 return True
#             elif play_again.lower() == 'no':
#                 return False
#             else:
#                 print("Invalid input. Please enter 'yes' or 'no.'")


# snake_game = SnakeGame()
# snake_game.start_game()


##########################################################################################level inside settings

# import os
# import random
# import keyboard
# import time
# import json
# from colorama import Fore, init

# class SnakeGame:
#     def __init__(self):
#         self.CELLS = []
#         self.snake_body = []
#         self.DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = ()
#         self.speed = 0.5  # Default speed
#         self.game_running = True
#         self.position_choice = 1  # Default starting position
#         self.difficulty_level = 1  # Default difficulty level

#     def print_field(self):
#         os.system('clear')
#         for row in range(self.FIELD_HEIGHT):
#             for col in range(self.FIELD_WIDTH):
#                 current_cell = (row, col)
#                 if current_cell in self.snake_body:
#                     if current_cell == self.snake_body[0]:  # Check if it's the head
#                         print(Fore.MAGENTA + 'X', end='')
#                     else:
#                         print(Fore.YELLOW + 'X', end='')
#                 elif current_cell == self.apple_pos:
#                     print(Fore.RED + '*', end='')
#                 elif col in (0, self.FIELD_WIDTH - 1) or row in (0, self.FIELD_HEIGHT - 1):
#                     print(Fore.CYAN + '#', end='')  # Boundaries without color
#                 else:
#                     print(' ', end='')

#             print()  # Move to the next line after printing each row

#     def place_apple(self):
#         col = random.randint(1, self.FIELD_WIDTH - 2)
#         row = random.randint(1, self.FIELD_HEIGHT - 2)
#         while (col, row) in self.snake_body:
#             col = random.randint(1, self.FIELD_WIDTH - 2)
#             row = random.randint(1, self.FIELD_HEIGHT - 2)
#         return (row, col)

#     def start_game(self):
#         while True:
#             print("Choose an option:")
#             print("1. New Game")
#             print("2. Resume Game")
#             print("3. Quit")
#             print("4. Help")
#             print("5. About")
#             print("6. Settings")
#             try:
#                 option_choice = int(input("Enter the number corresponding to your choice: "))
#                 if option_choice == 1:
#                     self.game_running = True
#                     while self.game_running:
#                         self.setup_game()
#                         self.set_difficulty()
#                         self.play_game()
#                     if not self.play_again():
#                         print("Thanks for playing. Goodbye!")
#                         break
#                 elif option_choice == 2:
#                     saved_game_file = 'saved_game.json'
#                     if os.path.exists(saved_game_file):
#                         self.load_game(saved_game_file)
#                         self.game_running = True
#                         self.play_game()
#                         if not self.play_again():
#                             print("Thanks for playing. Goodbye!")
#                             break
#                     else:
#                         print("No saved game found. Please start a new game.")
#                 elif option_choice == 3:
#                     print("Thanks for playing. Goodbye!")
#                     break
#                 elif option_choice == 4:
#                     self.display_help()
#                 elif option_choice == 5:
#                     self.display_about()
#                 elif option_choice == 6:
#                     self.configure_settings()
#                 else:
#                     print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def display_help(self):
#         print("Help:")
#         print("Use the following keys during the game:")
#         print("'w' - Move up")
#         print("'a' - Move left")
#         print("'s' - Move down")
#         print("'d' - Move right")
#         print("'q' - Quit the game")

#     def display_about(self):
#         print("About:")
#         print("This game was developed by Dinesh and Dhatchayani.")
#         print("Snake Game Version 1.0")
#         print("To understand the instructions of the game press 3.")

#     def configure_settings(self):
#         while True:
#             print("Settings:")
#             print("1. Snake position")
#             print("2. Difficulty level")
#             print("3. Back to main menu")
#             try:
#                 setting_choice = int(input("Enter the number corresponding to your choice: "))
#                 if setting_choice == 1:
#                     self.configure_snake_position()
#                 elif setting_choice == 2:
#                     self.configure_difficulty_level()
#                 elif setting_choice == 3:
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1, 2, or 3.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def configure_snake_position(self):
#         print("Choose snake starting position:")
#         print("1. (0, 0)")
#         print("2. Random position")
#         try:
#             self.position_choice = int(input("Enter the number corresponding to your choice: "))
#             if self.position_choice not in [1, 2]:
#                 print("Invalid choice. Please enter 1 or 2.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     def configure_difficulty_level(self):
#         print("Choose difficulty level:")
#         print("1. Easy")
#         print("2. Intermediate")
#         print("3. Hard")
#         try:
#             self.difficulty_level = int(input("Enter the number corresponding to your choice: "))
#             if self.difficulty_level not in [1, 2, 3]:
#                 print("Invalid choice. Please enter 1, 2, or 3.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     def setup_game(self):
#         while True:
#             print("Enter the width of the field")
#             try:
#                 self.FIELD_WIDTH = int(input())
#                 print("Enter the height of the field")
#                 try:
#                     self.FIELD_HEIGHT = int(input())
#                     if 7 <= self.FIELD_WIDTH <= 32 and 7 <= self.FIELD_HEIGHT <= 20:
#                         break
#                     else:
#                         print("Area dimensions should be between 7X7 and 32X20. Please try again.")
#                 except ValueError:
#                     print("Please enter a valid integer for the height.")
#             except ValueError:
#                 print("Please enter a valid integer for the width.")

#         self.CELLS = [(row, col) for row in range(self.FIELD_HEIGHT) for col in range(self.FIELD_WIDTH)]

#         # snake starting position
#         if self.position_choice == 1:
#             self.snake_body = [(1, 1)]
#         elif self.position_choice == 2:
#             valid_start_positions = [(row, col) for row in range(2, self.FIELD_HEIGHT - 2) for col in range(2, self.FIELD_WIDTH - 2)]
#             self.snake_body = [random.choice(valid_start_positions)]

#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = self.place_apple()

#         # Print the initial state
#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()
#         #it solves the overspeed problem of first few fields
#         time.sleep(1)

#     def save_game(self, file_path):
#         game_state = {
#             'snake_body': self.snake_body,
#             'direction': self.direction,
#             'eaten': self.eaten,
#             'apple_pos': self.apple_pos,
#             'speed': self.speed,
#             'game_running': self.game_running,
#             'FIELD_WIDTH': self.FIELD_WIDTH,
#             'FIELD_HEIGHT': self.FIELD_HEIGHT,
#             'position_choice': self.position_choice,
#             'difficulty_level': self.difficulty_level
#         }

#         with open(file_path, 'w') as file:
#             json.dump(game_state, file)

#     def load_game(self, file_path):
#         with open(file_path, 'r') as file:
#             game_state = json.load(file)

#         self.snake_body = game_state['snake_body']
#         self.direction = game_state['direction']
#         self.eaten = game_state['eaten']
#         self.speed = game_state['speed']
#         self.game_running = game_state['game_running']
#         self.FIELD_WIDTH = game_state['FIELD_WIDTH']
#         self.FIELD_HEIGHT = game_state['FIELD_HEIGHT']
#         self.position_choice = game_state['position_choice']
#         self.difficulty_level = game_state['difficulty_level']

#         # Place the apple using the current snake's body
#         self.apple_pos = self.place_apple()

#         os.system('clear')
#         print("\nGame loaded:")
#         self.print_field()

#     def set_difficulty(self):
#         if self.difficulty_level == 1:
#             self.speed = 0.5  # Set speed for Easy
#         elif self.difficulty_level == 2:
#             self.speed = 0.3  # Set speed for Intermediate
#         elif self.difficulty_level == 3:
#             self.speed = 0.1  # Set speed for Hard

#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def play_game(self):
#         total_apples_eaten = 0

#         def on_key_press(e):
#             if e.event_type == keyboard.KEY_DOWN:
#                 if e.name in ['w', 'up']:
#                     self.direction = self.DIRECTIONS['up']
#                 elif e.name in ['a', 'left']:
#                     self.direction = self.DIRECTIONS['left']
#                 elif e.name in ['s', 'down']:
#                     self.direction = self.DIRECTIONS['down']
#                 elif e.name in ['d', 'right']:
#                     self.direction = self.DIRECTIONS['right']
#                 elif e.name == 'q':
#                     self.game_running = False
#                 elif e.name == 'p':  # Save key
#                     self.save_game('saved_game.json')
#                 elif e.name == 'l':  # Load key
#                     self.load_game('saved_game.json')

#         keyboard.hook(on_key_press)

#         while self.game_running:
#             prev_snake_body = self.snake_body.copy()
#             new_head = (
#                 self.snake_body[0][0] + self.direction[0],
#                 self.snake_body[0][1] + self.direction[1]
#             )
#             self.snake_body.insert(0, new_head)
#             if not self.eaten:
#                 self.snake_body.pop(-1)
#             self.eaten = False

#             # check if snake eats the apple
#             if self.snake_body[0] == self.apple_pos:
#                 self.apple_pos = self.place_apple()
#                 self.eaten = True
#                 total_apples_eaten += 1

#             # check if the snake bites itself
#             if self.snake_body[0] in self.snake_body[1:]:
#                 os.system('clear')
#                 print("\nSnake bites itself. Game over.")
#                 self.game_running = False
#                 break

#             # check if the snakes move in the reverse direction
#             if len(prev_snake_body) > 1 and (
#                     new_head[0] == prev_snake_body[1][0] and new_head[1] == prev_snake_body[1][1]
#             ):
#                 os.system('clear')
#                 print("\nReverse motion. Game over.")
#                 self.game_running = False
#                 break

#             # check if the snake crosses the boundary
#             if (
#                     self.snake_body[0][0] in (0, self.FIELD_HEIGHT - 1)
#                     or self.snake_body[0][1] in (0, self.FIELD_WIDTH - 1)
#             ):
#                 os.system('clear')
#                 print("\nYou crossed the boundary. Game over.")
#                 self.game_running = False
#                 break

#             self.print_field()
#             time.sleep(self.speed)  # Use the selected speed for the delay

#         keyboard.unhook_all()
#         print("Total Apples Eaten:", total_apples_eaten)

#     def play_again(self):
#         while True:
#             play_again = input("Do you want to play again? (yes/no): ")
#             if play_again.lower() == 'yes':
#                 self.eaten = False
#                 return True
#             elif play_again.lower() == 'no':
#                 return False
#             else:
#                 print("Invalid input. Please enter 'yes' or 'no.'")


# snake_game = SnakeGame()
# snake_game.start_game()

#######################################################################################correcting apple and points

# import os
# import random
# import keyboard
# import time
# import json
# from colorama import Fore, init

# class SnakeGame:
#     def __init__(self):
#         self.CELLS = []
#         self.snake_body = []
#         self.DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = ()
#         self.speed = 0.5  # Default speed
#         self.game_running = True
#         self.position_choice = 1  # Default starting position
#         self.difficulty_level = 1  # Default difficulty level
#         self.total_apples_eaten = 0  # Initialize total apples eaten

#     def print_field(self):
#         os.system('clear')
#         for row in range(self.FIELD_HEIGHT):
#             for col in range(self.FIELD_WIDTH):
#                 current_cell = (row, col)
#                 if current_cell in self.snake_body:
#                     if current_cell == self.snake_body[0]:  # Check if it's the head
#                         print(Fore.MAGENTA + 'X', end='')
#                     else:
#                         print(Fore.YELLOW + 'X', end='')
#                 elif current_cell == self.apple_pos:
#                     print(Fore.RED + '*', end='')
#                 elif col in (0, self.FIELD_WIDTH - 1) or row in (0, self.FIELD_HEIGHT - 1):
#                     print(Fore.CYAN + '#', end='')  # Boundaries without color
#                 else:
#                     print(' ', end='')

#             print()  # Move to the next line after printing each row

#     def place_apple(self):
#         col = random.randint(1, self.FIELD_WIDTH - 2)
#         row = random.randint(1, self.FIELD_HEIGHT - 2)
#         while (col, row) in self.snake_body:
#             col = random.randint(1, self.FIELD_WIDTH - 2)
#             row = random.randint(1, self.FIELD_HEIGHT - 2)
#         return (row, col)

#     def start_game(self):
#         while True:
#             print("Choose an option:")
#             print("1. New Game")
#             print("2. Resume Game")
#             print("3. Quit")
#             print("4. Help")
#             print("5. About")
#             print("6. Settings")
#             try:
#                 option_choice = int(input("Enter the number corresponding to your choice: "))
#                 if option_choice == 1:
#                     self.game_running = True
#                     while self.game_running:
#                         self.setup_game()
#                         self.set_difficulty()
#                         self.play_game()
#                     if not self.play_again():
#                         print("Thanks for playing. Goodbye!")
#                         break
#                 elif option_choice == 2:
#                     saved_game_file = 'saved_game.json'
#                     if os.path.exists(saved_game_file):
#                         self.load_game(saved_game_file)
#                         self.game_running = True
#                         self.play_game()
#                         if not self.play_again():
#                             print("Thanks for playing. Goodbye!")
#                             break
#                     else:
#                         print("No saved game found. Please start a new game.")
#                 elif option_choice == 3:
#                     print("Thanks for playing. Goodbye!")
#                     break
#                 elif option_choice == 4:
#                     self.display_help()
#                 elif option_choice == 5:
#                     self.display_about()
#                 elif option_choice == 6:
#                     self.configure_settings()
#                 else:
#                     print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def display_help(self):
#         print("Help:")
#         print("Use the following keys during the game:")
#         print("'w' - Move up")
#         print("'a' - Move left")
#         print("'s' - Move down")
#         print("'d' - Move right")
#         print("'q' - Quit the game")

#     def display_about(self):
#         print("About:")
#         print("This game was developed by Dinesh and Dhatchayani.")
#         print("Snake Game Version 1.0")
#         print("To understand the instructions of the game press 3.")

#     def configure_settings(self):
#         while True:
#             print("Settings:")
#             print("1. Snake position")
#             print("2. Difficulty level")
#             print("3. Back to main menu")
#             try:
#                 setting_choice = int(input("Enter the number corresponding to your choice: "))
#                 if setting_choice == 1:
#                     self.configure_snake_position()
#                 elif setting_choice == 2:
#                     self.configure_difficulty_level()
#                 elif setting_choice == 3:
#                     break
#                 else:
#                     print("Invalid choice. Please enter 1, 2, or 3.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     def configure_snake_position(self):
#         print("Choose snake starting position:")
#         print("1. (0, 0)")
#         print("2. Random position")
#         try:
#             self.position_choice = int(input("Enter the number corresponding to your choice: "))
#             if self.position_choice not in [1, 2]:
#                 print("Invalid choice. Please enter 1 or 2.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     def configure_difficulty_level(self):
#         print("Choose difficulty level:")
#         print("1. Easy")
#         print("2. Intermediate")
#         print("3. Hard")
#         try:
#             self.difficulty_level = int(input("Enter the number corresponding to your choice: "))
#             if self.difficulty_level not in [1, 2, 3]:
#                 print("Invalid choice. Please enter 1, 2, or 3.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     def setup_game(self):
#         while True:
#             print("Enter the width of the field")
#             try:
#                 self.FIELD_WIDTH = int(input())
#                 print("Enter the height of the field")
#                 try:
#                     self.FIELD_HEIGHT = int(input())
#                     if 7 <= self.FIELD_WIDTH <= 32 and 7 <= self.FIELD_HEIGHT <= 20:
#                         break
#                     else:
#                         print("Area dimensions should be between 7X7 and 32X20. Please try again.")
#                 except ValueError:
#                     print("Please enter a valid integer for the height.")
#             except ValueError:
#                 print("Please enter a valid integer for the width.")

#         self.CELLS = [(row, col) for row in range(self.FIELD_HEIGHT) for col in range(self.FIELD_WIDTH)]

#         # snake starting position
#         if self.position_choice == 1:
#             self.snake_body = [(1, 1)]
#         elif self.position_choice == 2:
#             valid_start_positions = [(row, col) for row in range(2, self.FIELD_HEIGHT - 2) for col in range(2, self.FIELD_WIDTH - 2)]
#             self.snake_body = [random.choice(valid_start_positions)]

#         self.direction = self.DIRECTIONS['right']
#         self.eaten = False
#         self.apple_pos = self.place_apple()

#         # Print the initial state
#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()
#         time.sleep(1)

#     def save_game(self, file_path):
#         game_state = {
#             'snake_body': self.snake_body,
#             'direction': self.direction,
#             'eaten': self.eaten,
#             'apple_pos': self.apple_pos,
#             'speed': self.speed,
#             'game_running': self.game_running,
#             'FIELD_WIDTH': self.FIELD_WIDTH,
#             'FIELD_HEIGHT': self.FIELD_HEIGHT,
#             'position_choice': self.position_choice,
#             'difficulty_level': self.difficulty_level,
#             'total_apples_eaten': self.total_apples_eaten  # Include total apples eaten in the saved state
#         }

#         with open(file_path, 'w') as file:
#             json.dump(game_state, file)

#     def load_game(self, file_path):
#         with open(file_path, 'r') as file:
#             game_state = json.load(file)

#         self.snake_body = game_state['snake_body']
#         self.direction = game_state['direction']
#         self.eaten = game_state['eaten']
#         self.speed = game_state['speed']
#         self.game_running = game_state['game_running']
#         self.FIELD_WIDTH = game_state['FIELD_WIDTH']
#         self.FIELD_HEIGHT = game_state['FIELD_HEIGHT']
#         self.position_choice = game_state['position_choice']
#         self.difficulty_level = game_state['difficulty_level']
#         self.total_apples_eaten = game_state.get('total_apples_eaten', 0)  # Load total apples eaten

#         # Place the apple using the current snake's body
#         self.apple_pos = self.place_apple()

#         os.system('clear')
#         print("\nGame loaded:")
#         self.print_field()

#     def set_difficulty(self):
#         if self.difficulty_level == 1:
#             self.speed = 0.5  # Set speed for Easy
#         elif self.difficulty_level == 2:
#             self.speed = 0.3  # Set speed for Intermediate
#         elif self.difficulty_level == 3:
#             self.speed = 0.1  # Set speed for Hard

#         os.system('clear')
#         print("\nInitial State:")
#         self.print_field()

#     def play_game(self):
#         total_apples_eaten = self.total_apples_eaten

#         def on_key_press(e):
#             nonlocal total_apples_eaten
#             if e.event_type == keyboard.KEY_DOWN:
#                 if e.name in ['w', 'up']:
#                     self.direction = self.DIRECTIONS['up']
#                 elif e.name in ['a', 'left']:
#                     self.direction = self.DIRECTIONS['left']
#                 elif e.name in ['s', 'down']:
#                     self.direction = self.DIRECTIONS['down']
#                 elif e.name in ['d', 'right']:
#                     self.direction = self.DIRECTIONS['right']
#                 elif e.name == 'q':
#                     self.game_running = False
#                 elif e.name == 'p':  # Save key
#                     self.total_apples_eaten = total_apples_eaten
#                     self.save_game('saved_game.json')
#                 elif e.name == 'l':  # Load key
#                     self.load_game('saved_game.json')

#         keyboard.hook(on_key_press)

#         while self.game_running:
#             prev_snake_body = self.snake_body.copy()
#             new_head = (
#                 self.snake_body[0][0] + self.direction[0],
#                 self.snake_body[0][1] + self.direction[1]
#             )
#             self.snake_body.insert(0, new_head)
#             if not self.eaten:
#                 self.snake_body.pop(-1)
#             self.eaten = False

#             # check if snake eats the apple
#             if self.snake_body[0] == self.apple_pos:
#                 self.apple_pos = self.place_apple()
#                 self.eaten = True
#                 total_apples_eaten += 1

#             # check if the snake bites itself
#             if self.snake_body[0] in self.snake_body[1:]:
#                 os.system('clear')
#                 print("\nSnake bites itself. Game over.")
#                 self.game_running = False
#                 break

#             # check if the snakes move in the reverse direction
#             if len(prev_snake_body) > 1 and (
#                     new_head[0] == prev_snake_body[1][0] and new_head[1] == prev_snake_body[1][1]
#             ):
#                 os.system('clear')
#                 print("\nReverse motion. Game over.")
#                 self.game_running = False
#                 break

#             # check if the snake crosses the boundary
#             if (
#                     self.snake_body[0][0] in (0, self.FIELD_HEIGHT - 1)
#                     or self.snake_body[0][1] in (0, self.FIELD_WIDTH - 1)
#             ):
#                 os.system('clear')
#                 print("\nYou crossed the boundary. Game over.")
#                 self.game_running = False
#                 break

#             self.print_field()
#             time.sleep(self.speed)  # Use the selected speed for the delay

#         keyboard.unhook_all()
#         print("Total Apples Eaten:", total_apples_eaten)

#     def play_again(self):
#         while True:
#             play_again = input("Do you want to play again? (yes/no): ")
#             if play_again.lower() == 'yes':
#                 self.eaten = False
#                 return True
#             elif play_again.lower() == 'no':
#                 return False
#             else:
#                 print("Invalid input. Please enter 'yes' or 'no.'")


# snake_game = SnakeGame()
# snake_game.start_game()

#############################################################################################final pakka


import os
import random
import keyboard
import time
import json
from colorama import Fore, init

class SnakeGame:
    def __init__(self):
        self.CELLS = []
        self.snake_body = []
        self.DIRECTIONS = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
        self.direction = self.DIRECTIONS['right']
        self.eaten = False
        self.apple_pos = ()
        self.speed = 0.5  # Default speed
        self.game_running = True
        self.position_choice = 1  # Default starting position
        self.difficulty_level = 1  # Default difficulty level
        self.total_apples_eaten = 0  # Initialize total apples eaten

    def print_field(self):
        os.system('clear')
        for row in range(self.FIELD_HEIGHT):
            for col in range(self.FIELD_WIDTH):
                current_cell = (row, col)
                if current_cell in self.snake_body:
                    if current_cell == self.snake_body[0]:  # Check if it's the head
                        print(Fore.MAGENTA + 'X', end='')
                    else:
                        print(Fore.YELLOW + 'X', end='')
                elif current_cell == self.apple_pos:
                    print(Fore.RED + '*', end='')
                elif col in (0, self.FIELD_WIDTH - 1) or row in (0, self.FIELD_HEIGHT - 1):
                    print(Fore.CYAN + '#', end='')  # Boundaries without color
                else:
                    print(' ', end='')

            print()  # Move to the next line after printing each row

    def place_apple(self):
        col = random.randint(1, self.FIELD_WIDTH - 2)
        row = random.randint(1, self.FIELD_HEIGHT - 2)
        while (col, row) in self.snake_body:
            col = random.randint(1, self.FIELD_WIDTH - 2)
            row = random.randint(1, self.FIELD_HEIGHT - 2)
        return (row, col)

    def start_game(self):
        while True:
            print("Choose an option:")
            print("1. New Game")
            print("2. Resume Game")
            print("3. Quit")
            print("4. Help")
            print("5. About")
            print("6. Settings")
            try:
                option_choice = int(input("Enter the number corresponding to your choice: "))
                if option_choice == 1:
                    self.game_running = True
                    while self.game_running:
                        self.setup_game()
                        self.set_difficulty()
                        self.play_game()
                    if not self.play_again():
                        print("Thanks for playing. Goodbye!")
                        break
                elif option_choice == 2:
                    saved_game_file = 'saved_game.json'
                    if os.path.exists(saved_game_file):
                        self.load_game(saved_game_file)
                        self.game_running = True
                        self.play_game()
                        if not self.play_again():
                            print("Thanks for playing. Goodbye!")
                            break
                    else:
                        print("No saved game found. Please start a new game.")
                elif option_choice == 3:
                    print("Thanks for playing. Goodbye!")
                    break
                elif option_choice == 4:
                    self.display_help()
                elif option_choice == 5:
                    self.display_about()
                elif option_choice == 6:
                    self.configure_settings()
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def display_help(self):
        print("Help:")
        print("Use the following keys during the game:")
        print("'w' - Move up")
        print("'a' - Move left")
        print("'s' - Move down")
        print("'d' - Move right")
        print("'q' - Quit the game")

    def display_about(self):
        print("About:")
        print("This game was developed by Dinesh and Dhatchayani.")
        print("Snake Game Version 1.0")
        print("To understand the instructions of the game press 3.")

    def configure_settings(self):
        while True:
            print("Settings:")
            print("1. Snake position")
            print("2. Difficulty level")
            print("3. Back to main menu")
            try:
                setting_choice = int(input("Enter the number corresponding to your choice: "))
                if setting_choice == 1:
                    self.configure_snake_position()
                elif setting_choice == 2:
                    self.configure_difficulty_level()
                elif setting_choice == 3:
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def configure_snake_position(self):
        print("Choose snake starting position:")
        print("1. (0, 0)")
        print("2. Random position")
        try:
            self.position_choice = int(input("Enter the number corresponding to your choice: "))
            if self.position_choice not in [1, 2]:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def configure_difficulty_level(self):
        print("Choose difficulty level:")
        print("1. Easy")
        print("2. Intermediate")
        print("3. Hard")
        try:
            self.difficulty_level = int(input("Enter the number corresponding to your choice: "))
            if self.difficulty_level not in [1, 2, 3]:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def setup_game(self):
        while True:
            print("Enter the width of the field")
            try:
                self.FIELD_WIDTH = int(input())
                print("Enter the height of the field")
                try:
                    self.FIELD_HEIGHT = int(input())
                    if 7 <= self.FIELD_WIDTH <= 32 and 7 <= self.FIELD_HEIGHT <= 20:
                        break
                    else:
                        print("Area dimensions should be between 7X7 and 32X20. Please try again.")
                except ValueError:
                    print("Please enter a valid integer for the height.")
            except ValueError:
                print("Please enter a valid integer for the width.")

        self.CELLS = [(row, col) for row in range(self.FIELD_HEIGHT) for col in range(self.FIELD_WIDTH)]

        # snake starting position
        if self.position_choice == 1:
            self.snake_body = [(1, 1)]
        elif self.position_choice == 2:
            valid_start_positions = [(row, col) for row in range(2, self.FIELD_HEIGHT - 2) for col in range(2, self.FIELD_WIDTH - 2)]
            self.snake_body = [random.choice(valid_start_positions)]

        self.direction = self.DIRECTIONS['right']
        self.eaten = False
        self.apple_pos = self.place_apple()

        # Print the initial state
        os.system('clear')
        print("\nInitial State:")
        self.print_field()
        time.sleep(1)

    def save_game(self, file_path):
        game_state = {
            'snake_body': self.snake_body,
            'direction': self.direction,
            'eaten': self.eaten,
            'apple_pos': self.apple_pos,
            'speed': self.speed,
            'game_running': self.game_running,
            'FIELD_WIDTH': self.FIELD_WIDTH,
            'FIELD_HEIGHT': self.FIELD_HEIGHT,
            'position_choice': self.position_choice,
            'difficulty_level': self.difficulty_level,
            'total_apples_eaten': self.total_apples_eaten  # Include total apples eaten in the saved state
        }

        with open(file_path, 'w') as file:
            json.dump(game_state, file)

    def load_game(self, file_path):
        with open(file_path, 'r') as file:
            game_state = json.load(file)

        self.snake_body = game_state['snake_body']
        self.direction = game_state['direction']
        self.eaten = game_state['eaten']
        self.speed = game_state['speed']
        self.game_running = game_state['game_running']
        self.FIELD_WIDTH = game_state['FIELD_WIDTH']
        self.FIELD_HEIGHT = game_state['FIELD_HEIGHT']
        self.position_choice = game_state['position_choice']
        self.difficulty_level = game_state['difficulty_level']
        self.total_apples_eaten = game_state.get('total_apples_eaten', 0)  # Load total apples eaten

        # Place the apple using the current snake's body
        self.apple_pos = self.place_apple()

        os.system('clear')
        print("\nGame loaded:")
        self.print_field()

    def set_difficulty(self):
        if self.difficulty_level == 1:
            self.speed = 0.5  # Set speed for Easy
        elif self.difficulty_level == 2:
            self.speed = 0.3  # Set speed for Intermediate
        elif self.difficulty_level == 3:
            self.speed = 0.1  # Set speed for Hard

        os.system('clear')
        print("\nInitial State:")
        self.print_field()

    def play_game(self):
        total_apples_eaten = self.total_apples_eaten

        def on_key_press(e):
            nonlocal total_apples_eaten
            if e.event_type == keyboard.KEY_DOWN:
                if e.name in ['w', 'up']:
                    self.direction = self.DIRECTIONS['up']
                elif e.name in ['a', 'left']:
                    self.direction = self.DIRECTIONS['left']
                elif e.name in ['s', 'down']:
                    self.direction = self.DIRECTIONS['down']
                elif e.name in ['d', 'right']:
                    self.direction = self.DIRECTIONS['right']
                elif e.name == 'q':
                    self.game_running = False
                elif e.name == 'p':  # Save key
                    self.total_apples_eaten = total_apples_eaten
                    self.save_game('saved_game.json')
                elif e.name == 'l':  # Load key
                    self.load_game('saved_game.json')

        keyboard.hook(on_key_press)

        while self.game_running:
            prev_snake_body = self.snake_body.copy()
            new_head = (
                self.snake_body[0][0] + self.direction[0],
                self.snake_body[0][1] + self.direction[1]
            )
            self.snake_body.insert(0, new_head)
            if not self.eaten:
                self.snake_body.pop(-1)
            self.eaten = False

            # check if snake eats the apple
            if self.snake_body[0] == self.apple_pos:
                self.apple_pos = self.place_apple()
                self.eaten = True
                total_apples_eaten += 1

            # check if the snake bites itself
            if self.snake_body[0] in self.snake_body[1:]:
                os.system('clear')
                print(Fore.WHITE + "\nSnake bites itself. Game over.")
                self.game_running = False
                break

            # check if the snakes move in the reverse direction
            if len(prev_snake_body) > 1 and (
                    new_head[0] == prev_snake_body[1][0] and new_head[1] == prev_snake_body[1][1]
            ):
                os.system('clear')
                print(Fore.WHITE + "\nReverse motion. Game over.")
                self.game_running = False
                break

            # check if the snake crosses the boundary
            if (
                    self.snake_body[0][0] in (0, self.FIELD_HEIGHT - 1)
                    or self.snake_body[0][1] in (0, self.FIELD_WIDTH - 1)
            ):
                os.system('clear')
                print(Fore.WHITE + "\nYou crossed the boundary. Game over.")
                self.game_running = False
                break

            self.print_field()
            time.sleep(self.speed)  # Use the selected speed for the delay

        keyboard.unhook_all()
        print(Fore.WHITE + "Total Apples Eaten:", total_apples_eaten)

    def play_again(self):
        while True:
            play_again = input(Fore.WHITE + "Do you want to play again? (yes/no): ")
            if play_again.lower() == 'yes':
                self.eaten = False
                return True
            elif play_again.lower() == 'no':
                return False
            else:
                print("Invalid input. Please enter 'yes' or 'no.'")


snake_game = SnakeGame()
snake_game.start_game()
