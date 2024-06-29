import os, re
import numpy as np
from itertools import groupby
from chess import Board
from chess.engine import SimpleEngine, Limit
import multiprocessing

# Encoding for pieces relative to me and my opponent.
square_states = {
    'empty':0,
    'my_pawn':1, 'my_passant_pawn':2,
    'my_virgin_rook':3, 'my_moved_rook':4,
    'my_knight':5, 'my_ls_bishop':6, 'my_ds_bishop':7,
    'my_queen':8, 'my_virgin_king':9, 'my_moved_king':10,
    'op_pawn':11, 'op_passant_pawn':12,
    'op_virgin_rook':13, 'op_moved_rook':14,
    'op_knight':15, 'op_ls_bishop':16, 'op_ds_bishop':17,
    'op_queen':18, 'op_virgin_king':19, 'op_moved_king':20
}

# Mappers to help convert from rank/file to array index based on white/black perspective.
white_filemap = {v:k for k,v in dict(enumerate('abcdefgh')).items()}
black_filemap = {v:k for k,v in dict(enumerate('hgfedcba')).items()}
white_rankmap = {rank:8-int(rank) for rank in range(1,9)}
black_rankmap = {rank:int(rank)-1 for rank in range(1,9)}
board_map = {
    'rank': {'white': white_rankmap, 'black': black_rankmap},
    'file': {'white': white_filemap, 'black': black_filemap}
}

# Mapper from algrbraic notation to piece index
piece_map = {'P': {1, 2}, 'R': {3, 4}, 'N': {5}, 'B': {6, 7}, 'Q': {8}, 'K': {9, 10}}

def init_board(play_as='white'):
    '''Initializes board for new game.'''
    if play_as == 'white': # Queen - King
        my_pieces = ['my_virgin_rook', 'my_knight', 'my_ds_bishop', 'my_queen', 'my_virgin_king', 'my_ls_bishop', 'my_knight', 'my_virgin_rook']
        op_pieces = ['op_virgin_rook', 'op_knight', 'op_ls_bishop', 'op_queen', 'op_virgin_king', 'op_ds_bishop', 'op_knight', 'op_virgin_rook']
    elif play_as == 'black': # King - Queen
        my_pieces = ['my_virgin_rook', 'my_knight', 'my_ds_bishop', 'my_virgin_king', 'my_queen', 'my_ls_bishop', 'my_knight', 'my_virgin_rook']
        op_pieces = ['op_virgin_rook', 'op_knight', 'op_ls_bishop', 'op_virgin_king', 'op_queen', 'op_ds_bishop', 'op_knight', 'op_virgin_rook']
    board = np.zeros((8,8), dtype=int) # Initialize Board
    board[-2] = np.array([1]*8, dtype=int) # My Pawns
    board[1] = np.array([11]*8, dtype=int) # Op Pawns
    board[-1] = np.array([square_states[p] for p in my_pieces], dtype=int) # My pieces
    board[0] = np.array([square_states[p] for p in op_pieces], dtype=int) # Op pieces
    return board

def conjugate_board(board):
    '''Swaps board perspective to opponent. Will be useful for self-play.'''
    conj_board = np.array(board, copy=True)
    conj_board = np.flip(conj_board)
    return np.where(conj_board >= 11, conj_board - 10, np.where(conj_board >= 1, conj_board + 10, 0))

###:: COMPUTING AVAILABLE MOVES ::###

def on_board(r, c):
    '''Returns true if given row / column is on the board.'''
    return (0 <= r <= 7) & (0 <= c <= 7)

def is_light_square(r, c):
    '''Returns true if the row / column position is a light square.'''
    # Satisfy yourself that this is invariant to player perspective.
    return ((r + c) % 2) == 0

def pawn_moves(board):
    pawn_locs = np.transpose(((board == 1) | (board == 2)).nonzero())
    candidate_boards = []
    
    # Scan for available pawn moves
    for r,c in pawn_locs:

        # Can move forward one? 
        # rp: proposed position row, cp: proposed position column
        rp, cp = r-1, c
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] == 0): # Forward square clear
                if rp == 0: # Must convert pawn when landing on the back rank
                    # If we create a bishop, it is a light square or dark square bishop?
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)
                else:
                    candidate = board.copy()
                    candidate[rp,cp] = 1
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)

        # Forward two?
        rp, cp = r-2, c
        if r == 6: # Pawn not yet moved
            if (board[rp+1, cp] == 0): # First forward square clear
                if (board[rp, cp] == 0): # Second forward square clear
                    candidate = board.copy()
                    candidate[rp,cp] = 2
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)

        # Forward left?
        rp, cp = r-1, c-1
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] >= 11): # Forward left square enemy-populated? Capture.
                if rp == 0: # Must promote pawn if landing on back rank
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)

                else:
                    candidate = board.copy()
                    candidate[rp,cp] = 1
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
  
        # Forward-left en-passant. Needs game moderator to enforce lapsing of en-passant option.
        rp, cp = r-1, c-1
        if on_board(rp, cp):
            if (board[r, cp] == 12): # Adjacent square occupied by enemy passant pawn.
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate[r,cp] = 0 # Enemy passant pawn captured.
                candidate_boards.append(candidate)

        # Forward right?
        rp, cp = r-1, c+1
        if on_board(rp, cp): # Proposed position is on the board
            if (board[rp, cp] >= 11): # Forward right square enemy-populated? Capture.
                if rp == 0: # Must promote pawn if landing on back rank
                    bishop = 6 if is_light_square(rp,cp) else 7
                    for piece in [4,5,bishop,8]:
                        candidate = board.copy()
                        candidate[rp,cp] = piece
                        candidate[r,c] = 0
                        candidate_boards.append(candidate)
                else:
                    candidate = board.copy()
                    candidate[rp,cp] = 1
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    
                        
        # Forward-right en-passant. Needs game moderator to enforce lapsing of en-passant option.
        rp, cp = r-1, c+1
        if on_board(rp, cp):
            if (board[r, cp] == 12): # Adjacent square occupied by enemy passant pawn.
                candidate = board.copy()
                candidate[rp,cp] = 1
                candidate[r,c] = 0
                candidate[r,cp] = 0 # Enemy passant pawn captured.
                candidate_boards.append(candidate)
                        
    return candidate_boards

def rook_moves(board):
    rook_locs = np.transpose(((board==3) | (board==4)).nonzero())
    candidate_boards = []

    for r,c in rook_locs:

        # Move left?
        offset = 1
        while True:
            rp, cp = r, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break
                else:
                    break
            else:
                break

        # Move right?
        offset = 1
        while True:
            rp, cp = r, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move forward?
        offset = 1
        while True:
            rp, cp = r-offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward?
        offset = 1
        while True:
            rp, cp = r+offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 4 # Becomes moved rook
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break
    
    return candidate_boards

def castle_moves(board):
    rook_locs = np.transpose(((board==3) | (board==4)).nonzero())
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    kr, kc = king_loc
    
    candidate_boards = []

    for r,c in rook_locs:
        if board[r,c] == 3: # Must be rook's first move
            if board[kr,kc] == 9: # Must be king's first move
                rng = sorted([c,kc]) # Rook and King columns in sorted order
                rng = range(rng[0]+1,rng[1]) # Range over intermediate columns
                if all([board[7,ci] == 0 for ci in rng]): # Path must be clear of other pieces
                    # Is the enemy attacking any of the path squares?
                    enemy_board = conjugate_board(board) # Swap to their perspective
                    # Enemy moves considered include all but castle, which can otherwise trigger an infinite recursion. Enemy castle cannot attack back rank anyway.
                    enemy_moves = pawn_moves(enemy_board) + rook_moves(enemy_board) + knight_moves(enemy_board) + bishop_moves(enemy_board) + queen_moves(enemy_board) + king_moves(enemy_board)
                    enemy_moves = [conjugate_board(em) for em in enemy_moves] # Conjugate back to our perspective

                    if c == 0: # Castle left?
                        if all([all([enm_move[7,ci] == 0 for ci in [kc-1, kc-2]]) for enm_move in enemy_moves]):
                            candidate = board.copy()
                            candidate[7,kc-2] = 10 # Move King
                            candidate[7,kc] = 0
                            candidate[7,kc-1] = 4 # Move rook
                            candidate[7,0] = 0
                            candidate_boards.append(candidate)

                    elif c == 7: # Castle right?
                        if all([all([enm_move[7,ci] == 0 for ci in [kc+1, kc+2]]) for enm_move in enemy_moves]):
                            candidate = board.copy()
                            candidate[7,kc+2] = 10 # Move King
                            candidate[7,kc] = 0
                            candidate[7,kc+1] = 4 # Move rook
                            candidate[7,7] = 0
                            candidate_boards.append(candidate)
                            
    return candidate_boards

def knight_moves(board):
    knight_locs = np.transpose((board==5).nonzero())
    candidate_boards = []

    for r,c in knight_locs:

        proposals = [ # All the relative positions the rook could move to if allowed.
            (r+2,c+1),(r+1,c+2),
            (r-2,c+1),(r-1,c+2),
            (r+2,c-1),(r+1,c-2),
            (r-2,c-1),(r-1,c-2),
        ]

        for rp,cp in proposals:
            if on_board(rp, cp):
                if (board[rp, cp] == 0) or (board[rp, cp] >= 11): # Proposed position clear or enemy-occupied
                    candidate = board.copy()
                    candidate[rp,cp] = 5 # Move knight
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)

    return candidate_boards

def bishop_moves(board):
    bishop_locs = np.transpose(((board==6) | (board==7)).nonzero())
    candidate_boards = []

    for r,c in bishop_locs:
        bishop = 6 if is_light_square(r,c) else 7 # Which one?

        # Move forward-left?
        offset = 1
        while True:
            rp, cp = r-offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move forward-right?
        offset = 1
        while True:
            rp, cp = r-offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward-left?
        offset = 1
        while True:
            rp, cp = r+offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move backward-right?
        offset = 1
        while True:
            rp, cp = r+offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = bishop # Move bishop
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break
                
    return candidate_boards

def queen_moves(board):
    queen_locs = np.transpose((board==8).nonzero())
    candidate_boards = []
    
    for r,c in queen_locs:

        # Move left?
        offset = 1
        while True:
            rp, cp = r, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break

        # Move right?
        offset = 1
        while True:
            rp, cp = r, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move forward?
        offset = 1
        while True:
            rp, cp = r-offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward?
        offset = 1
        while True:
            rp, cp = r+offset, c
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search
                
        # Move forward-left?
        offset = 1
        while True:
            rp, cp = r-offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move forward-right?
        offset = 1
        while True:
            rp, cp = r-offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward-left?
        offset = 1
        while True:
            rp, cp = r+offset, c-offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search

        # Move backward-right?
        offset = 1
        while True:
            rp, cp = r+offset, c+offset
            if on_board(rp, cp):
                if board[rp, cp] == 0: # Proposed position clear
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    offset += 1 # Increment offset
                elif board[rp, cp] >= 11: # Can take opposition piece
                    candidate = board.copy()
                    candidate[rp,cp] = 8 # Move queen
                    candidate[r,c] = 0
                    candidate_boards.append(candidate)
                    break # End search
                else:
                    break # End search
            else:
                break # End search
                
    return candidate_boards

def king_moves(board):
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    candidate_boards = []
    
    r,c = king_loc
    proposals = [
        (r+1,c),(r-1,c),
        (r,c+1),(r,c-1),
        (r+1,c+1),(r-1,c-1),
        (r+1,c-1),(r-1,c+1),
    ]
    
    for rp,cp in proposals:
        if on_board(rp, cp):
            if (board[rp, cp] == 0) or (board[rp, cp] >= 11): # Proposed position clear or enemy-occupied
                candidate = board.copy()
                candidate[rp,cp] = 10 # Move King
                candidate[r,c] = 0
                candidate_boards.append(candidate)
                
    return candidate_boards

def candidate_moves(board):
    '''Return candidate boards representing each potential move from a given board configuration.'''
    candidates = pawn_moves(board) + rook_moves(board) + knight_moves(board) + bishop_moves(board) + queen_moves(board) + king_moves(board) + castle_moves(board)
    return [c for c in candidates if not in_check(c)]

def in_check(board):
    '''Returns true if I am in check. Can then filter available moves for those that get me out of check. If none, checkmate.'''
    # Where is my King?
    king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
    # Is the enemy attacking any of the path squares?
    enemy_board = conjugate_board(board) # Swap to their perspective
    # Enemy moves considered include all but castle, which can otherwise trigger an infinite recursion. Enemy castle cannot attack king anyway.
    enemy_moves = pawn_moves(enemy_board) + rook_moves(enemy_board) + knight_moves(enemy_board) + bishop_moves(enemy_board) + queen_moves(enemy_board) + king_moves(enemy_board)
    enemy_moves = [conjugate_board(em) for em in enemy_moves] # Conjugate back to our perspective
    # Not the case that where my king is currently located, the piece in that location is no longer my king, for possible enemy moves.
    return any([em[tuple(king_loc)] not in piece_map['K'] for em in enemy_moves]) # i.e. none of my enemy's moves can replace my king.

def is_draw(board):
    '''Returns true if board state is a known draw due to insufficient material for check-mate.'''
    # King - King
    K_K = set(board[np.where(board>0)]).issubset({9,10,19,20})
    # King - King | Knight
    K_KN = set(board[np.where(board>0)]).issubset({9,10,19,20,5})
    # King - King | Bishop
    K_KB = (
    set(board[np.where(board>0)]).issubset({9,10,19,20,6}) or
    set(board[np.where(board>0)]).issubset({9,10,19,20,7}) 
    )
    # King | Bishop - King | Bishop (bishops on opposite colors) 
    KB_KB = (
        set(board[np.where(board>0)]).issubset({9,10,19,20,6,17}) or
        set(board[np.where(board>0)]).issubset({9,10,19,20,7,16})
    )
    return (K_K or K_KN or K_KB or KB_KB)

def check_gameover(board, n_candidates, n_moves, turn, max_moves=np.inf):
    # If there are no candidate moves available, that could mean checkmate or stalemate.
    if n_candidates == 0:
        if in_check(board): # I am in check and have no moves available. Checkmate, opponent wins.
            scores = (-1.0, 1.0) # Current player loses, opponent wins; (-1, 1)
            return scores # return game results
        else: # I am not in check but have no availabe moves. Stalemate. Draw.
            scores = (0.0, 0.0)
            return scores # return game results
    elif is_draw(board): # Draw.
        scores = (0.0, 0.0)
        return scores # return game results
    elif n_moves >= max_moves: # max moves reached
        return (0.0, 0.0)
    else: # Return False, game is not over. 
        return False

def get_played(board, move, turn, candidates, announce_outcome=False):
    # Init dummies for castle and promotion
    castle_dest = None
    promote_to = None

    # Detect pawn promotion
    if '=' in move:
        promote_to = move[-1]
        move = move[:-2]

    # A few special cases to handle
    if move == '1-0':
        if announce_outcome:
            print('White wins.')
        return None

    elif move == '0-1':
        if announce_outcome:
            print('Black wins.')
        return None

    elif move == '1/2-1/2' or move == '1/2-1/2':
        if announce_outcome:
            print('Draw.')
        return None

    elif move == 'O-O': # Castle king side
        if turn == 'white':
            king_dest = (7,6)
            rook_dest = (7,5)
            castle_dest = [king_dest, rook_dest]
        elif turn == 'black':
            king_dest = (7,1)
            rook_dest = (7,2)
            castle_dest = [king_dest, rook_dest]
        else:
            raise Exception('Failed to resolve turn for king-side castle.')

    elif move == 'O-O-O': # Castle queen side
        if turn == 'white':
            king_dest = (7,2)
            rook_dest = (7,3)
            castle_dest = [king_dest, rook_dest]
        elif turn == 'black':
            king_dest = (7,5)
            rook_dest = (7,4)
            castle_dest = [king_dest, rook_dest]
        else:
            raise Exception('Failed to resolve turn for queen-side castle.')

    # Piece move - analyse the move description for piece origin and destination information
    else:
        if move[0].islower():
            piece = 'P'
            orig = move[:-2]
        else:
            piece = move[0]
            orig = move[1:-2]

        # Destination
        dest = move[-2:]
        file, rank = dest
        r_dest = board_map['rank'][turn][int(rank)]
        c_dest = board_map['file'][turn][file]

        # Origin cell if specified
        if len(orig) == 0:
            r_orig = None
            c_orig = None

        elif len(orig) == 1:

            if orig.isnumeric():
                rank = int(orig)
                r_orig = board_map['rank'][turn][rank]
                c_orig = None
            else:
                file = orig
                c_orig = board_map['file'][turn][file]
                r_orig = None

        elif len(orig) == 2:

            file, rank = orig
            r_orig = board_map['rank'][turn][rank]
            c_orig = board_map['file'][turn][file]

    # Filter boards based on king/rook destinations
    if castle_dest: # Castle dest not None only if move passed in was a castle move.
        king_dest, rook_dest = castle_dest
        played = [c for c in candidates if c[king_dest]==10 and c[rook_dest]==4]

    else:
        # Filter boards based on destination cell containing the correct piece
        if promote_to:
            # Destination square must match the promoted piece
            played = [c for c in candidates if c[r_dest, c_dest] in piece_map[promote_to]]
            if c_orig:
                played = [c for c in played if not np.all(c[:,c_orig]==board[:,c_orig])]
            if r_orig:
                played = [c for c in played if not np.all(c[r_orig,:]==board[r_orig,:])]
            # Must be one less pawn on the board as promoted.
            played = [c for c in played if (c==1).sum() == (board==1).sum() - 1]
            # Must be one more promoted piece on the board as promoted.
            played = [c for c in played if sum([(c==prom).sum() for prom in piece_map[promote_to]]) == sum([(board==prom).sum() for prom in piece_map[promote_to]]) + 1]

        else:
            # Destination square must match the moving piece
            played = [c for c in candidates if c[r_dest, c_dest] in piece_map[piece]]
            # Must be same number of this kind of piece on the board
            played = [c for c in played if sum([(c==pind).sum() for pind in piece_map[piece]]) == sum([(board==pind).sum() for pind in piece_map[piece]])]

            # If knight and row origin specified, must be one fewer knight in that row after
            if piece == "N" and r_orig:
                board_count = np.array([p in piece_map[piece] for p in board[r_orig,:]]).sum()
                played = [c for c in played if np.array([p in piece_map[piece] for p in c[r_orig,:]]).sum() < board_count]

            # If knight and col origin specified, must be one fewer knight in that col after
            if piece == "N" and c_orig:
                board_count = np.array([p in piece_map[piece] for p in board[:,c_orig]]).sum()
                played = [c for c in played if np.array([p in piece_map[piece] for p in c[:,c_orig]]).sum() < board_count]

        if piece == 'R': # Also check here that if it's a rook moving, that the king hasn't moved; not a castle move.
            king_loc = np.transpose(((board==9) | (board==10)).nonzero())[0] # Only ever one
            played = [c for c in played if c[tuple(king_loc)] == board[tuple(king_loc)]]

        # If there are still more than one plausible board, filter boards based on origin
        if len(played) > 1:
            if not str(r_orig).isnumeric(): # Origin row implied
                r_origs = np.concatenate([list((board[:,c_orig]==i).nonzero()) for i in piece_map[piece]], axis=1).flatten()

                if len(r_origs) > 1:
                    played = [c for c in played if not np.all(c[:,c_orig] == board[:,c_orig])]
                    # if the col origin = col dest then
                    if c_orig == c_dest:
                        # there should be the same number of this kind of piece in that column
                        played = [c for c in played if sum([(c[:,c_orig]==i).sum() for i in piece_map[piece]]) == sum([(board[:,c_orig]==i).sum() for i in piece_map[piece]])]
                else:
                    r_orig = r_origs.item()
                    played = [c for c in played if c[r_orig, c_orig] == 0]

            elif not str(c_orig).isnumeric(): # Origin column implied
                # Potential column origins
                c_origs = np.concatenate([list((board[r_orig,:]==i).nonzero()) for i in piece_map[piece]], axis=1).flatten()

                if len(c_origs) > 1:
                    played = [c for c in played if not np.all(c[r_orig,:] == board[r_orig,:])]
                    # If the row origin == row dest then 
                    if r_orig == r_dest:
                        # there should be the same number of this kind of piece on that row
                        played = [c for c in played if sum([(c[r_orig,:]==i).sum() for i in piece_map[piece]]) == sum([(board[r_orig,:]==i).sum() for i in piece_map[piece]])]
                else:
                    c_orig = c_origs.item()
                    played = [c for c in played if c[r_orig, c_orig] == 0]

            else: # Origin fully specified
                assert str(r_orig).isnumeric() and str(c_orig).isnumeric() # Origin fully specified
                played = [c for c in played if c[r_orig, c_orig] == 0]

    # Logic above should have identified the one move that was played.
    assert len(played) == 1, f'''
    ERROR: turn: {turn} move: {move} piece: {piece} promote_to: {promote_to}\n\n{board}\n\n{played}\n\n
    ({r_orig} {c_orig} {r_dest} {c_dest})
    '''
    return played[0]

def which_board(board, candidates):
    index = [i for i,candidate in enumerate(candidates) if np.all(candidate == board)]
    assert len(index) == 1, "ERROR: NONE OR MORE THAN ONE MATCH?"
    return index[0]

def play_historic(board, chs_board, moves, turn, announce_outcome=False):

    # Output of function
    states = []

    # Loop over provided moves
    for move_ind, move in enumerate(moves):

        # Retrieve the move
        move = move.replace('x','').replace('+','').replace('#','') # Drop capture/check/checkmate notation - unnecessary
        # Revert any passant pawns to regular pawns if they survived the enemy's turn.
        board[board==2] = 1
        # Calculate legal moves from this position
        candidates = candidate_moves(board)
        # Generate candidates and identify the board of the move that was made
        played_board = get_played(board, move, turn, candidates, announce_outcome)
        # Maybe terminate
        if played_board is None:
            return states

        # Idenfity the index of the played board among the list of candidates
        played_index = which_board(played_board, candidates)

        # Get fen for each candidate?
        fens = [None] * len(candidates)
        for cand_move in chs_board.legal_moves:
            cand_pgn = chs_board.san(cand_move)
            cand_pgn = cand_pgn.replace('x','').replace('+','').replace('#','')
            target_board = get_played(board, cand_pgn, turn, candidates, announce_outcome)
            cand_ind = which_board(target_board, candidates)
            chs_board_fwd = chs_board.copy()
            # So we push candidate move to our hypothetical board.
            chs_board_fwd.push_san(cand_pgn)
            # We have now played, it is our opponents turn to play.
            # So stockfish will eval the fen for this board from the perspective of our opponent!
            # So to get a score for our position, we invert the sign of our opponents positional score here.
            fens[cand_ind] = chs_board_fwd.fen()

        # Somehow we found a move the chess engine discounted? Maybe duplicate? Ignore.
        candidates = [c for c,f in zip(candidates, fens) if f is not None]
        fens = [f for f in fens if f is not None]

        # Accelerate with multiprocessing
        with multiprocessing.Pool(os.cpu_count()) as pool:
            stock_evals = list(pool.map(evaluate_position, fens))

        # invert the sign of our opponents positional score here.
        stock_evals = [-1.0 * s for s in stock_evals]

        # Save candidates and played. Maybe also save elo to be able to train a model that is parameterised by strength?
        states.append({
            "move": move_ind, 
            "turn": turn, 
            "board": board, 
            "candidates": candidates, 
            "fens": fens,
            "evaluations": stock_evals,
            "played_index": played_index
        })

        # Update the commercial board state as well
        chs_board.push_san(move)
        # Record the new board state
        board = played_board
        # Conjugate board to other side's view
        board = conjugate_board(board)
        # Turn play over to the other side.
        if turn == 'white':
            turn = 'black'
        else:
            turn = 'white'

def evaluate_position(fen, time_limit=10, depth_limit=25, STOCKFISH_PATH="/root/chess-hackathon/stockfish/stockfish"):
    # Create a chess board from the FEN string
    board = Board(fen)
    # Initialize the Stockfish engine
    with SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Perform the evaluation to max depth or time limit.
        info = engine.analyse(board, Limit(depth=depth_limit, time=time_limit))
        # Extract the score
        score = info['score'].relative.score(mate_score=10_000)
    return score
