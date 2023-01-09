import numpy as np
import random


class NQUEENS(object):
    """
        ....Q.....
        .....Q....
        Q.........
        ........Q.
        .......Q..
        ...Q......
        .........Q
        .Q........
        ..Q.......
        ......Q...
        
        This class represents a way to model in python the nqueen problem.
        The problem is to determine a configuration of n queens
        on a nxn chessboard such that no queen can be taken by
        one another. In this version, each queens is assigned
        to one column, and only one queen can be on each line.
        The evaluation function therefore only counts the number
        of conflicts along the diagonals.

    """

    def __init__(self, size):
        self.size = size # Board is size x size


    def set_nqueens(self,permutation):
        """
        Knowing that each queen is assigned to a single colum, we use a permutaion to place
        queens on the rows. This ensure that a single queen is put on each row
        """
        self.board = np.identity(self.size,dtype=np.int)
        self.board = self.board[permutation,:]
        return self.count_mutual_conflicts()


    def count_mutual_conflicts(self):
        """
        Count the total number of conflicts on the board
        Conflicts represents queens on the same diagonal
        -----
        Return number of conflicts
        """
        total_conflicts=0
        rotated_board = np.fliplr(self.board)
        for i in range(-1*(self.size-1),self.size):
            c_fd = np.sum(np.diagonal(self.board,i))
            c_sd = np.sum(np.diagonal(rotated_board,i))
            total_conflicts += (c_fd*(c_fd-1))/2
            total_conflicts += (c_sd*(c_sd-1))/2
        return total_conflicts


    def random_nqueens(self):
        """
        Place randomly the queens on the board
        Becareful, positions may be in conflicts
        ----
        Return a list of positions
        """
        row_indexes = list(range(self.size))
        random.shuffle(row_indexes)
        self.board = np.identity(self.size,dtype=np.int)
        self.board = self.board[row_indexes,:]

    def show_board(self):
        """
        ASCII board 
        """
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                if self.board[i,j] == 1:
                    line += "Q"
                else:
                    line += "."
            print(line)


if __name__ == "__main__":
    prob = NQUEENS(20)
    prob.random_nqueens()
    prob.show_board()
    for i in range(10):
        indexes = list(range(20))
        random.shuffle(indexes)
        print(prob.set_nqueens(indexes))

        



