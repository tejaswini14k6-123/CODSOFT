"""Task 2: Tic-Tac-Toe AI with Minimax Algorithm
Implements an AI player that uses the Minimax algorithm to play Tic-Tac-Toe unbeatable.
"""

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.human = 'O'
        self.ai = 'X'
    
    def print_board(self):
        print('\n')
        for i in range(3):
            print(f' {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ')
            if i < 2:
                print('-----------')
        print('\n')
    
    def is_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        return any(all(self.board[i] == player for i in combo) for combo in win_combinations)
    
    def is_board_full(self):
        return ' ' not in self.board
    
    def get_empty_cells(self):
        return [i for i in range(9) if self.board[i] == ' ']
    
    def minimax(self, depth, is_maximizing):
        """Minimax algorithm to find the best move"""
        if self.is_winner(self.ai):
            return 10 - depth
        if self.is_winner(self.human):
            return depth - 10
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for cell in self.get_empty_cells():
                self.board[cell] = self.ai
                score = self.minimax(depth + 1, False)
                self.board[cell] = ' '
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for cell in self.get_empty_cells():
                self.board[cell] = self.human
                score = self.minimax(depth + 1, True)
                self.board[cell] = ' '
                best_score = min(score, best_score)
            return best_score
    
    def get_best_move(self):
        """Find the best move for AI using minimax"""
        best_score = float('-inf')
        best_move = None
        for cell in self.get_empty_cells():
            self.board[cell] = self.ai
            score = self.minimax(0, False)
            self.board[cell] = ' '
            if score > best_score:
                best_score = score
                best_move = cell
        return best_move
    
    def play(self):
        print("Welcome to Tic-Tac-Toe! You are O, AI is X")
        print("Positions are numbered 0-8:")
        print(" 0 | 1 | 2")
        print("-----------")
        print(" 3 | 4 | 5")
        print("-----------")
        print(" 6 | 7 | 8")
        
        while True:
            self.print_board()
            
            # Human move
            while True:
                try:
                    position = int(input("Your move (0-8): "))
                    if position < 0 or position > 8 or self.board[position] != ' ':
                        print("Invalid move!")
                        continue
                    self.board[position] = self.human
                    break
                except ValueError:
                    print("Please enter a number between 0 and 8")
            
            if self.is_winner(self.human):
                self.print_board()
                print("You win!")
                break
            
            if self.is_board_full():
                self.print_board()
                print("It's a draw!")
                break
            
            # AI move
            print("AI is thinking...")
            ai_move = self.get_best_move()
            self.board[ai_move] = self.ai
            print(f"AI chose position {ai_move}")
            
            if self.is_winner(self.ai):
                self.print_board()
                print("AI wins!")
                break
            
            if self.is_board_full():
                self.print_board()
                print("It's a draw!")
                break

if __name__ == "__main__":
    game = TicTacToe()
    game.play()
