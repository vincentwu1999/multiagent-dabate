import random, chess, chess.pgn
from typing import List, Dict, Any

random.seed(7)

def load_chess_positions_for_move_validity(k: int = 50) -> List[Dict[str, Any]]:
    positions = []
    for _ in range(k):
        board = chess.Board()
        for __ in range(random.randint(6, 12)):
            moves = list(board.legal_moves)
            if not moves: break
            board.push(random.choice(moves))
        own = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == board.turn]
        if not own: continue
        origin = random.choice(own)
        origin_alg = chess.square_name(origin)
        if not [m for m in board.legal_moves if m.from_square == origin]:
            continue
        game = chess.pgn.Game.from_board(board)
        positions.append({"pgn": str(game), "origin": origin_alg, "board_fen": board.fen()})
    return positions
