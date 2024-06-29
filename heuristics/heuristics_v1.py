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
    - Material count: weighted material count
    - Mobility count: unweighted material count

    Numbers are higher if the position is better for you.
    """

    @staticmethod
    def material_count(board: torch.Tensor) -> torch.Tensor:
        # TODO(Adriano) is the king and passed king really zero?
        # TODO(Adriano) are these values ok? Should we have multiple versions?
        # TODO(Adriano) we do not take into account the material quality
        # https://en.wikipedia.org/wiki/Chess_piece_relative_value
        piece_values = torch.tensor(
            [
                0,  # Empty
                1,  # Pawn
                1,  # Passant Pawn
                5,  # Unmoved Rook
                5,  # Moved Rook
                3,  # Knight
                3,  # Light Bishop
                3,  # Dark Bishop
                9,  # Queen
                0,  # Unmoved King
                0,  # Moved King
            ]
        )
        my_material = torch.sum(piece_values[board[board <= 10]])
        opponent_material = torch.sum(piece_values[board[board > 10] - OPPONENT_OFFSET])
        val = torch.tensor([my_material - opponent_material], dtype=torch.float32)
        assert val.shape == (1,)
        return val

    @staticmethod
    def mobility_count(board: torch.Tensor) -> torch.Tensor:
        my_pieces = torch.sum(board <= 10)
        opp_pieces = torch.sum(board > 10)
        return torch.tensor([my_pieces - opp_pieces], dtype=torch.float32)

    @staticmethod
    def pawn_majority(board: torch.Tensor) -> torch.Tensor:
        # Get each of our queenside and kingside
        my_queenside = torch.sum((board == PAWN) | (board == PASSANT_PAWN))[:, :3]
        my_kingside = torch.sum((board == PAWN) | (board == PASSANT_PAWN))[:, 5:]
        opp_queenside = torch.sum(
            (board == PAWN + OPPONENT_OFFSET)
            | (board == PASSANT_PAWN + OPPONENT_OFFSET)
        )[:, :3]
        opp_kingside = torch.sum(
            (board == PAWN + OPPONENT_OFFSET)
            | (board == PASSANT_PAWN + OPPONENT_OFFSET)
        )[:, 5:]

        my_majority = (my_queenside > opp_queenside).float() + (
            my_kingside > opp_kingside
        ).float()
        opp_majority = (opp_queenside > my_queenside).float() + (
            opp_kingside > my_kingside
        ).float()

        return torch.tensor([my_majority - opp_majority], dtype=torch.float32)


class PieceTableValuation:
    """
    For each type of piece we have one or more ways of measuring its
    value based on its position, independently of the other pieces.
    These are calculated from tables of values (that often emphasize centrality
    or other good properties).

    These return numbers that are higher the better they are for you.
    """

    def pawn_and_knight_tables(board: torch.Tensor) -> torch.Tensor:
        """
        Value pawns to be better in the center and worse on the outsides. The
        same is done for knights, but is more aggressively done for knights.
        """
        assert board.shape == (8, 8)
        # fmt: off
        pawn_table = torch.tensor([
            [0,  0,  0,  0,  0,  0,  0,  0,],
            [50, 50, 50, 50, 50, 50, 50, 50,],
            [10, 10, 20, 30, 30, 20, 10, 10,],
            [5,  5, 10, 25, 25, 10,  5,  5,],
            [0,  0,  0, 20, 20,  0,  0,  0,],
            # TODO(Adriano) lower means closer to enemy
            # that seems incorrect since having pawns that are closer to queening seems
            # to be quite good. Why is this?
            [5, -5,-10,  0,  0,-10, -5,  5,],
            [5, 10, 10,-20,-20, 10, 10,  5,],
            [0,  0,  0,  0,  0,  0,  0,  0],
        ], dtype=torch.float32)
        # fmt: on
        assert pawn_table.shape == (8, 8)

        # fmt: off
        knight_table = torch.tensor([
            [-50,-40,-30,-30,-30,-30,-40,-50,],
            [-40,-20,  0,  0,  0,  0,-20,-40,],
            [-30,  0, 10, 15, 15, 10,  0,-30,],
            [-30,  5, 15, 20, 20, 15,  5,-30,],
            [-30,  0, 15, 20, 20, 15,  0,-30,],
            [-30,  5, 10, 15, 15, 10,  5,-30,],
            [-40,-20,  0,  5,  5,  0,-20,-40,],
            [-50,-40,-30,-30,-30,-30,-40,-50],
        ], dtype=torch.float32)
        # fmt: on
        assert knight_table.shape == (8, 8)

        score = 0
        flat_board = board.flatten()
        flat_pawn_table = pawn_table.flatten()
        flat_knight_table = knight_table.flatten()
        for i in range(64):
            piece = flat_board[i]
            # Pawn
            if piece == PAWN:
                score += flat_pawn_table[i]
            # Enemy Pawn
            elif piece == PAWN + OPPONENT_OFFSET:
                # TODO(Adriano) this is technically wrong but should not matter
                # (seems to flip both dimensions instead of only one)
                score -= flat_pawn_table[63 - i]
            # Knight
            elif piece == KNIGHT:
                score += flat_knight_table[i]
            # Enemy Knight
            elif piece == KNIGHT + OPPONENT_OFFSET:
                score -= flat_knight_table[63 - i]

        out = torch.tensor([score], dtype=torch.float32)
        assert out.shape == (1,)
        return out


class PawnStructure:
    """
    Calculates ABSOLUTE SCORES for pawn structure based on whether they are isolated or doubled.
    These are both bad things and we return a number (as usual) that is higher, the better the
    state (for you).
    """

    @staticmethod
    def __file_pawn_counts(board: torch.Tensor, offset: int) -> torch.Tensor:
        out = torch.sum(
            (board == PAWN + offset) | (board == PASSANT_PAWN + offset), dim=0
        )
        return out

    @staticmethod
    def __doubled_pawns_individual(
        board: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        file_my_pawns = PawnStructure.__file_pawn_counts(board, offset)
        doubled_pawns_total = torch.clamp(file_my_pawns - 1, min=0).sum().item()
        assert doubled_pawns_total <= 7
        assert doubled_pawns_total >= 0

    @staticmethod
    def __doubled_pawns(board: torch.Tensor) -> torch.Tensor:
        my_doubled_pawns = PawnStructure.__doubled_pawns_individual(board)
        opp_doubled_pawns = PawnStructure.__doubled_pawns_individual(
            board, OPPONENT_OFFSET
        )
        # Negative because this is BAD
        out = -torch.tensor([my_doubled_pawns - opp_doubled_pawns], dtype=torch.float32)
        return out

    @staticmethod
    def __isolated_pawns_individual(
        board: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        file_my_pawns = PawnStructure.__file_pawn_counts(board, offset)
        pawns_on_file = file_my_pawns > 0
        pawn_is_isolated = torch.zeros(8, dtype=torch.bool)
        for i in range(8):
            has_a_neighbor = False
            has_neighbors = has_neighbors or (i != 0 and pawns_on_file[i - 1])
            has_neighbors = has_neighbors or (i != 7 and pawns_on_file[i + 1])
            pawn_is_isolated[i] = not has_neighbors
        return torch.sum(pawn_is_isolated).item()

    @staticmethod
    def __isolated_pawns(board: torch.Tensor) -> torch.Tensor:
        my_isolated_pawns = PawnStructure.__isolated_pawns_individual(board)
        opp_isolated_pawns = PawnStructure.__isolated_pawns_individual(
            board, OPPONENT_OFFSET
        )
        # Negative because this is BAD
        out = -torch.tensor(
            [my_isolated_pawns - opp_isolated_pawns], dtype=torch.float32
        )
        return out

    @staticmethod
    def pawn_structure(board: torch.Tensor) -> torch.Tensor:
        """
        Value pawns based on whether they are doubled or isolated. We simply subtract
        the "position" for us and that for our opponent.
        """
        return torch.mean(
            [
                # Each of these is a measure of how much better we are than the opponent
                # by taking the negative of how much worse we are
                PawnStructure.__doubled_pawns(board),
                PawnStructure.__isolated_pawns(board),
            ]
        )

    @staticmethod
    def __passed_pawns_individual(board: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Return a COUNT of how many pawns we have that are passed. More is better.
        """
        assert offset in [0, OPPONENT_OFFSET]
        my_pawns = (board == PAWN + offset) | (board == PASSANT_PAWN + offset)
        opp_pawns = (board == PAWN + OPPONENT_OFFSET) | (
            board == PASSANT_PAWN + OPPONENT_OFFSET
        )
        # TODO(Adriano) need a test to make sure these sort of reversal iteration is OK
        opp_latest_per_row = torch.ones(8, dtype=torch.int32) * 9
        start, end, d, cmp = 0, 8, 1, max
        if offset == OPPONENT_OFFSET:
            start, end, d, cmp = 7, -1, -1, min
        for i in range(8):
            for j in range(start, end, d):
                if opp_pawns[i, j]:
                    opp_latest_per_row[i] = cmp(opp_latest_per_row[i], j)
        passed: int = 0
        for i in range(8):
            for j in range(opp_latest_per_row[i], end, d):
                if my_pawns[i, j]:
                    passed += 1
        return passed

    @staticmethod
    def passed_pawns(board: torch.Tensor) -> torch.Tensor:
        """
        Value pawns based on whether they are passed. A passed pawn is passed if
        there are no opponent pawns above it.
        """
        # NOTE: passed pawns are GOOD which is why we are taking the POSITIVE side
        out = torch.tensor(
            [
                PawnStructure.__passed_pawns_individual(board)
                - PawnStructure.__passed_pawns_individual(board, offset=OPPONENT_OFFSET)
            ],
            dtype=torch.float32,
        )
        assert out.shape == (1,)
        return out


class Centrality:
    """
    Centrality measures check that you control key zones of the board.
    - centrality_center_n: checks that the nxn center square is controlled
    - centrality_tempo_n: checks that the center n ranks are controlled

    This is always a difference between yours and the opponents'. Moreover,
    larger number is better for you.
    """

    @staticmethod
    def __centrality_center(board: torch.Tensor, left: int, right: int) -> torch.Tensor:
        center = board[left:right, left:right]
        my_control = torch.sum(center <= 10)
        opp_control = torch.sum(center > 10)
        return torch.tensor([my_control - opp_control], dtype=torch.float32)

    @staticmethod
    def centrality_center_2(board: torch.Tensor) -> torch.Tensor:
        return Centrality.__centrality_center(board, 3, 5)

    @staticmethod
    def centrality_center_4(board: torch.Tensor) -> torch.Tensor:
        return Centrality.__centrality_center(board, 2, 6)

    @staticmethod
    def __centrality_tempo(board: torch.Tensor, left: int, right: int) -> torch.Tensor:
        my_developed = torch.sum(board[left:right, :] <= 10)
        opp_developed = torch.sum(board[left:right, :] > 10)
        return torch.tensor([my_developed - opp_developed], dtype=torch.float32)

    @staticmethod
    def centrality_tempo_2(board: torch.Tensor) -> torch.Tensor:
        return Centrality.__centrality_tempo(board, 3, 5)

    @staticmethod
    def centrality_tempo_4(board: torch.Tensor) -> torch.Tensor:
        return Centrality.__centrality_tempo(board, 2, 6)


class GoodPieces:
    """
    Valuation based on good pieces like bishops, rooks, knights.
    """

    @staticmethod
    def bishop_pair(board: torch.Tensor) -> torch.Tensor:
        my_bishops = torch.sum((board == LIGHT_BISHOP) | (board == DARK_BISHOP))
        opp_bishops = torch.sum(
            (board == LIGHT_BISHOP + OPPONENT_OFFSET)
            | (board == DARK_BISHOP + OPPONENT_OFFSET)
        )
        my_pair = 1 if my_bishops >= 2 else 0
        opp_pair = 1 if opp_bishops >= 2 else 0
        return torch.tensor([my_pair - opp_pair], dtype=torch.float32)

    @staticmethod
    def rooks_on_open_files(board: torch.Tensor) -> torch.Tensor:
        # TODO(Adriano) does NOT take into account whether the rooks may have passed
        # the pawns.
        my_rooks = torch.sum((board == UNMOVED_ROOK) | (board == MOVED_ROOK), dim=0)
        opp_rooks = torch.sum(
            (board == UNMOVED_ROOK + OPPONENT_OFFSET)
            | (board == MOVED_ROOK + OPPONENT_OFFSET),
            dim=0,
        )
        pawns = torch.sum(
            (board == PAWN)
            | (board == PASSANT_PAWN)
            | (board == PAWN + OPPONENT_OFFSET)
            | (board == PASSANT_PAWN + OPPONENT_OFFSET),
            dim=0,
        )
        open_files = (pawns == 0).float()
        return torch.tensor(
            [torch.sum(my_rooks * open_files) - torch.sum(opp_rooks * open_files)],
            dtype=torch.float32,
        )


class King:
    """
    Valuation based on king states.
    """

    # TODO(Adriano) I don't fully understand this, in part because I'm a little
    # confused, for example, as to whether the board goes from top to bottom or vv.
    # NOTE: it shouldn't matter if this is correct, so long as it is either very much
    # high for good, or very much low for good.
    @staticmethod
    def __king_shield(board: torch.Tensor, king_pos: int, is_opponent: bool):
        # Seems to look just for pawns of the right type
        rank, file = king_pos // 8, king_pos % 8
        shield_score = 0
        pawn_type = PAWN + (OPPONENT_OFFSET if is_opponent else 0)
        # TODO(Adriano) why is it file + 2 and not file + 1?
        for f in range(max(0, file - 1), min(8, file + 2)):
            pawn_rank = rank + (-1 if is_opponent else 1)
            if 0 <= pawn_rank < 8 and board[pawn_rank, f] == pawn_type:
                shield_score += 1
        return shield_score

    @staticmethod
    def king_safety(board: torch.Tensor) -> torch.Tensor:
        my_king_pos = torch.where((board == UNMOVED_KING) | (board == MOVED_KING))
        opp_king_pos = torch.where(
            (board == UNMOVED_KING + OPPONENT_OFFSET)
            | (board == MOVED_KING + OPPONENT_OFFSET)
        )

        my_safety = King.__king_shield(
            board, my_king_pos[0][0] * 8 + my_king_pos[1][0], False
        )
        opp_safety = King.__king_shield(
            board, opp_king_pos[0][0] * 8 + opp_king_pos[1][0], True
        )

        return torch.tensor([my_safety - opp_safety], dtype=torch.float32)

    @staticmethod
    def king_tropism(board: torch.Tensor) -> torch.Tensor:
        my_king_pos = torch.where((board == UNMOVED_KING) | (board == MOVED_KING))
        opp_king_pos = torch.where(
            (board == UNMOVED_KING + OPPONENT_OFFSET)
            | (board == MOVED_KING + OPPONENT_OFFSET)
        )

        center = torch.tensor([3.5, 3.5])
        # Appears to be manhattan difference form the very center
        # TODO(Adriano) why is it bad to be near the center? I guess your king is less
        # safe in likelihood
        my_distance = torch.sum(
            torch.abs(
                torch.tensor([my_king_pos[0][0], my_king_pos[1][0]]).float() - center
            )
        )
        opp_distance = torch.sum(
            torch.abs(
                torch.tensor([opp_king_pos[0][0], opp_king_pos[1][0]]).float() - center
            )
        )

        return torch.tensor([opp_distance - my_distance], dtype=torch.float32)


################################ EXPORTS ################################
# The functions below here are generally what you'll want to use
def evaluate_position(board: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            StandardMaterialValuation.material_count(board),
            StandardMaterialValuation.mobility_count(board),
            StandardMaterialValuation.pawn_majority(board),
            PieceTableValuation.pawn_and_knight_tables(board),
            PawnStructure.pawn_structure(board),
            PawnStructure.passed_pawns(board),
            Centrality.centrality_center_2(board),
            Centrality.centrality_center_4(board),
            Centrality.centrality_tempo_2(board),
            Centrality.centrality_tempo_4(board),
            GoodPieces.bishop_pair(board),
            GoodPieces.rooks_on_open_files(board),
            King.king_safety(board),
            King.king_tropism(board),
        ]
    )
