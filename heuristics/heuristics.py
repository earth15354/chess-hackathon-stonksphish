from __future__ import annotations

###
### Thanks to Claude 3.5 Sonnet
###

import torch
import torch.nn.functional as F

# Constants
EMPTY = 0
PAWN, PASSANT_PAWN, UNMOVED_ROOK, MOVED_ROOK, KNIGHT = 1, 2, 3, 4, 5
LIGHT_BISHOP, DARK_BISHOP, QUEEN, UNMOVED_KING, MOVED_KING = 6, 7, 8, 9, 10
OPPONENT_OFFSET = 10

################################ UTILS LIB ################################
def is_my_piece(piece: int) -> bool:
    return 1 <= piece <= 10

def is_opponent_piece(piece: int) -> bool:
    return 11 <= piece <= 20

def piece_type(piece: int) -> int:
    return piece if is_my_piece(piece) else piece - OPPONENT_OFFSET

################################ HEURISTICS LIB ################################
class StandardMaterialValuation:
    """
    Standard material evaluation.
    """
    def material_count(board: torch.Tensor) -> torch.Tensor:
        # TODO(Adriano) is the king and passed king really zero?
        # TODO(Adriano) are these values ok? Should we have multiple versions?
        # TODO(Adriano) we do not take into account the material quality
        # https://en.wikipedia.org/wiki/Chess_piece_relative_value
        piece_values = torch.tensor([
            0, # Empty
            1, # Pawn
            1, # Passant Pawn
            5, # Unmoved Rook
            5, # Moved Rook
            3, # Knight
            3, # Light Bishop
            3, # Dark Bishop
            9, # Queen
            0, # Unmoved King
            0 # Moved King
        ])
        my_material = torch.sum(piece_values[board[board <= 10]])
        opponent_material = torch.sum(piece_values[board[board > 10] - OPPONENT_OFFSET])
        val = torch.tensor([my_material - opponent_material], dtype=torch.float32)
        assert val.shape == (1,)
        return val

class PieceTableValuation:
    """
    For each type of piece we have one or more ways of measuring its
    value based on its position, independently of the other pieces.
    These are calculated from tables of values (that often emphasize centrality
    or other good properties).
    """
    def pawn_and_knight_tables(board: torch.Tensor) -> torch.Tensor:
        pawn_table = torch.tensor([
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ], dtype=torch.float32)

        knight_table = torch.tensor([
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ], dtype=torch.float32)

        score = 0
        for i in range(64):
            piece = board.flatten()[i]
            # Pawn
            if piece == PAWN:
                score += pawn_table[i]
            # Enemy Pawn
            elif piece == PAWN + OPPONENT_OFFSET:
                score -= pawn_table[63 - i]
            elif piece == KNIGHT:
                score += knight_table[i]
            elif piece == KNIGHT + OPPONENT_OFFSET:
                score -= knight_table[63 - i]

        return torch.tensor([score], dtype=torch.float32)

def pawn_structure(board: torch.Tensor) -> torch.Tensor:
    doubled_pawns = 0
    isolated_pawns = 0
    
    for file in range(8):
        my_pawns = torch.sum((board == PAWN) | (board == PASSANT_PAWN), dim=0)[file]
        opp_pawns = torch.sum((board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET), dim=0)[file]
        
        if my_pawns > 1:
            doubled_pawns += my_pawns - 1
        if opp_pawns > 1:
            doubled_pawns -= opp_pawns - 1
        
        if my_pawns > 0 and (file == 0 or torch.sum((board == PAWN) | (board == PASSANT_PAWN), dim=0)[file-1] == 0) and \
           (file == 7 or torch.sum((board == PAWN) | (board == PASSANT_PAWN), dim=0)[file+1] == 0):
            isolated_pawns += 1
        if opp_pawns > 0 and (file == 0 or torch.sum((board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET), dim=0)[file-1] == 0) and \
           (file == 7 or torch.sum((board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET), dim=0)[file+1] == 0):
            isolated_pawns -= 1
    
    return torch.tensor([doubled_pawns * -0.5 + isolated_pawns * -0.5], dtype=torch.float32)

def king_safety(board: torch.Tensor) -> torch.Tensor:
    def king_shield(king_pos, is_opponent):
        rank, file = king_pos // 8, king_pos % 8
        shield_score = 0
        pawn_type = PAWN + (OPPONENT_OFFSET if is_opponent else 0)
        for f in range(max(0, file - 1), min(8, file + 2)):
            pawn_rank = rank + (-1 if is_opponent else 1)
            if 0 <= pawn_rank < 8 and board[pawn_rank, f] == pawn_type:
                shield_score += 1
        return shield_score

    my_king_pos = torch.where((board == UNMOVED_KING) | (board == MOVED_KING))
    opp_king_pos = torch.where((board == UNMOVED_KING + OPPONENT_OFFSET) | (board == MOVED_KING + OPPONENT_OFFSET))
    
    my_safety = king_shield(my_king_pos[0][0] * 8 + my_king_pos[1][0], False)
    opp_safety = king_shield(opp_king_pos[0][0] * 8 + opp_king_pos[1][0], True)
    
    return torch.tensor([my_safety - opp_safety], dtype=torch.float32)

def mobility(board: torch.Tensor) -> torch.Tensor:
    # This is a simplified mobility heuristic
    my_pieces = torch.sum(board <= 10)
    opp_pieces = torch.sum(board > 10)
    return torch.tensor([my_pieces - opp_pieces], dtype=torch.float32)

def control_of_center(board: torch.Tensor) -> torch.Tensor:
    center = board[3:5, 3:5]
    my_control = torch.sum(center <= 10)
    opp_control = torch.sum(center > 10)
    return torch.tensor([my_control - opp_control], dtype=torch.float32)

def bishop_pair(board: torch.Tensor) -> torch.Tensor:
    my_bishops = torch.sum((board == LIGHT_BISHOP) | (board == DARK_BISHOP))
    opp_bishops = torch.sum((board == LIGHT_BISHOP + OPPONENT_OFFSET) | (board == DARK_BISHOP + OPPONENT_OFFSET))
    my_pair = 1 if my_bishops >= 2 else 0
    opp_pair = 1 if opp_bishops >= 2 else 0
    return torch.tensor([my_pair - opp_pair], dtype=torch.float32)

def rooks_on_open_files(board: torch.Tensor) -> torch.Tensor:
    my_rooks = torch.sum((board == UNMOVED_ROOK) | (board == MOVED_ROOK), dim=0)
    opp_rooks = torch.sum((board == UNMOVED_ROOK + OPPONENT_OFFSET) | (board == MOVED_ROOK + OPPONENT_OFFSET), dim=0)
    pawns = torch.sum((board == PAWN) | (board == PASSANT_PAWN) | 
                      (board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET), dim=0)
    open_files = (pawns == 0).float()
    return torch.tensor([torch.sum(my_rooks * open_files) - torch.sum(opp_rooks * open_files)], dtype=torch.float32)

def passed_pawns(board: torch.Tensor) -> torch.Tensor:
    my_pawns = (board == PAWN) | (board == PASSANT_PAWN)
    opp_pawns = (board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET)
    
    my_passed = torch.sum(torch.all(torch.cumsum(opp_pawns.flip(0), dim=0).flip(0) == 0, dim=0) & my_pawns.any(dim=0))
    opp_passed = torch.sum(torch.all(torch.cumsum(my_pawns, dim=0) == 0, dim=0) & opp_pawns.any(dim=0))
    
    return torch.tensor([my_passed - opp_passed], dtype=torch.float32)

def piece_coordination(board: torch.Tensor) -> torch.Tensor:
    # Simplified coordination measure
    my_pieces = torch.sum(board <= 10)
    opp_pieces = torch.sum(board > 10)
    return torch.tensor([my_pieces - opp_pieces], dtype=torch.float32)

def tempo(board: torch.Tensor) -> torch.Tensor:
    my_developed = torch.sum(board[2:6, :] <= 10)
    opp_developed = torch.sum(board[2:6, :] > 10)
    return torch.tensor([my_developed - opp_developed], dtype=torch.float32)

def trapped_pieces(board: torch.Tensor) -> torch.Tensor:
    # Simplified trapped pieces heuristic
    my_trapped = torch.sum((board == KNIGHT) & (F.pad(board, (1, 1, 1, 1)) > 0))
    opp_trapped = torch.sum((board == KNIGHT + OPPONENT_OFFSET) & (F.pad(board, (1, 1, 1, 1)) > 0))
    return torch.tensor([opp_trapped - my_trapped], dtype=torch.float32)

def pawn_majority(board: torch.Tensor) -> torch.Tensor:
    my_queenside = torch.sum((board == PAWN) | (board == PASSANT_PAWN))[:, :3]
    my_kingside = torch.sum((board == PAWN) | (board == PASSANT_PAWN))[:, 5:]
    opp_queenside = torch.sum((board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET))[:, :3]
    opp_kingside = torch.sum((board == PAWN + OPPONENT_OFFSET) | (board == PASSANT_PAWN + OPPONENT_OFFSET))[:, 5:]
    
    my_majority = (my_queenside > opp_queenside).float() + (my_kingside > opp_kingside).float()
    opp_majority = (opp_queenside > my_queenside).float() + (opp_kingside > my_kingside).float()
    
    return torch.tensor([my_majority - opp_majority], dtype=torch.float32)

def king_tropism(board: torch.Tensor) -> torch.Tensor:
    my_king_pos = torch.where((board == UNMOVED_KING) | (board == MOVED_KING))
    opp_king_pos = torch.where((board == UNMOVED_KING + OPPONENT_OFFSET) | (board == MOVED_KING + OPPONENT_OFFSET))
    
    center = torch.tensor([3.5, 3.5])
    my_distance = torch.sum(torch.abs(torch.tensor([my_king_pos[0][0], my_king_pos[1][0]]).float() - center))
    opp_distance = torch.sum(torch.abs(torch.tensor([opp_king_pos[0][0], opp_king_pos[1][0]]).float() - center))
    
    return torch.tensor([opp_distance - my_distance], dtype=torch.float32)

def material_imbalance(board: torch.Tensor) -> torch.Tensor:
    piece_values = torch.tensor([0, 1, 1, 5, 5, 3, 3, 3, 9, 0, 0])
    my_material = torch.bincount(board[board <= 10].flatten(), minlength=11)
    opp_material = torch.bincount(board[board > 10].flatten() - OPPONENT_OFFSET, minlength=11)
    
    imbalance_score = 0.0
    # Bishop pair bonus
    if my_material[LIGHT_BISHOP] >= 1 and my_material[DARK_BISHOP] >= 1:
        imbalance_score += 0.5
    if opp_material[LIGHT_BISHOP] >= 1 and opp_material[DARK_BISHOP] >= 1:
        imbalance_score -= 0.5
    
    # Knight pair bonus
    if my_material[KNIGHT] >= 2:
        imbalance_score += 0.5
    if opp_material[KNIGHT] >= 2:
        imbalance_score -= 0.5
    
    # Rook pair bonus
    if my_material[UNMOVED_ROOK] + my_material[MOVED_ROOK] >= 2:
        imbalance_score += 0.5
    if opp_material[UNMOVED_ROOK] + opp_material[MOVED_ROOK] >= 2:
        imbalance_score -= 0.5
    
    # Queen vs. Two Rooks
    if my_material[QUEEN] == 1 and opp_material[UNMOVED_ROOK] + opp_material[MOVED_ROOK] == 2 and opp_material[QUEEN] == 0:
        imbalance_score += 0.5
    if opp_material[QUEEN] == 1 and my_material[UNMOVED_ROOK] + my_material[MOVED_ROOK] == 2 and my_material[QUEEN] == 0:
        imbalance_score -= 0.5
    
    return torch.tensor([imbalance_score], dtype=torch.float32)

################################ EXPORTS ################################
# The functions below here are generally what you'll want to use
def evaluate_position(board: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        material_count(board),
        piece_square_tables(board),
        pawn_structure(board),
        king_safety(board),
        mobility(board),
        control_of_center(board),
        bishop_pair(board),
        rooks_on_open_files(board),
        passed_pawns(board),
        piece_coordination(board),
        tempo(board),
        trapped_pieces(board),
        pawn_majority(board),
        king_tropism(board),
        material_imbalance(board),
    ])

