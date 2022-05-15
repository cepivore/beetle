import chess, chess.pgn, time, math, io
import numpy as np 
import chess.polyglot

infinity = 1000000
movecount = 0
MAX_TIME_MS = 10000

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

class Heuristics:
    PAWN_TABLE = np.array([
          0,  0,  0,  0,  0,  0,  0,  0,
         15, 15, 15,-20,-20, 15, 15, 15,
         -5, -5,-10, 10, 10,-10, -5, -5,
          0,  0,  0, 20, 20,  0,  0,  0,
          5,  5, 10, 25, 25, 10,  5,  5,
         10, 10, 20, 30, 30, 20, 10, 10,
         50, 50, 50, 50, 50, 50, 50, 50,
          0,  0,  0,  0,  0,  0,  0,  0
    ])
    KNIGHT_TABLE = np.array([
         -50, -40, -30, -30, -30, -30, -40, -50,
         -40, -20,   0,   5,   5,   0, -20, -40,
         -30,   5,  10,  15,  15,  10,   5, -30,
         -30,   0,  15,  20,  20,  15,   0, -30,
         -30,   0,  15,  20,  20,  15,   0, -30,
         -30,   0,  10,  15,  15,  10,   0, -30,
         -40, -20,   0,   0,   0,   0, -20, -40,
         -50, -40, -30, -30, -30, -30, -40, -50
    ])
    BISHOP_TABLE = np.array([
         -20, -10, -10, -10, -10, -10, -10, -20,
         -10,   5,   0,   0,   0,   0,   5, -10,
         -10,  10,  10,  10,  10,  10,  10, -10,
         -10,   0,  10,  10,  10,  10,   0, -10,
         -10,   5,   5,  10,  10,   5,   5, -10,
         -10,   0,   5,  10,  10,   5,   0, -10,
         -10,   0,   0,   0,   0,   0,   0, -10,
         -20, -10, -10, -10, -10, -10, -10, -20
    ])
    ROOK_TABLE = np.array([
          0,  0,  0,  5,  5,  0,  0,  0,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          5, 10, 10, 10, 10, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0
    ])
    QUEEN_TABLE = np.array([
         -20, -10, -10, -5, -5, -10, -10, -20,
         -10,   0,   5,  0,  0,   0,   0, -10,
         -10,   5,   5,  5,  5,   5,   0, -10,
           0,   0,   5,  5,  5,   5,   0,  -5,
          -5,   0,   5,  5,  5,   5,   0,  -5,
         -10,   0,   5,  5,  5,   5,   0, -10,
         -10,   0,   0,  0,  0,   0,   0, -10,
         -20, -10, -10, -5, -5, -10, -10, -20
    ])
    KING_TABLE = np.array([
          40,  40,   0,  0,  0,   0,  40,  40,
          -5,  15,  -5, -5, -5,  -5,  15,  -5,
         -10, -10, -10,-10,-10, -10, -10, -10,
         -10, -10, -10,-10,-10, -10, -10, -10,
         -10, -10, -10,-10,-10, -10, -10, -10,
         -10, -10, -10,-10,-10, -10, -10, -10,
         -10, -10, -10,-10,-10, -10, -10, -10,
         -10, -10, -10,-10,-10, -10, -10, -10
    ])

    @staticmethod
    def evaluate(board):
        pawns = get_pos_val(board, chess.PAWN, Heuristics.PAWN_TABLE)
        knights = get_pos_val(board, chess.KNIGHT, Heuristics.KNIGHT_TABLE)
        bishops = get_pos_val(board, chess.BISHOP, Heuristics.BISHOP_TABLE)
        rooks = get_pos_val(board, chess.ROOK, Heuristics.ROOK_TABLE)
        queens = get_pos_val(board, chess.QUEEN, Heuristics.QUEEN_TABLE)
        king = get_pos_val(board, chess.KING, Heuristics.KING_TABLE)

        score = pawns + knights + bishops + rooks + queens + king
        if board.turn == chess.WHITE:
            return score
        return -score

def get_piece_val(PieceType):
    value = 0
    if PieceType == chess.PAWN:
        value = 100
    if PieceType == chess.KNIGHT:
        value = 325
    if PieceType == chess.BISHOP:
        value = 330
    if PieceType == chess.ROOK:
        value = 500
    if PieceType == chess.QUEEN:
        value = 1000
    if PieceType == chess.KING:
        value = 100000
    return value

def negamax(board, depth, alpha, beta):
    global timems
    global start_time
    global ply
    best_score = -infinity
    incheck = board.is_check()
    legal = 0

    if is_time_limit_reached():
        return 0
    
    if incheck:
        depth += 1

    if depth==0:
        return qSearch(board, alpha, beta)
    
    for move in board.legal_moves:
        board.push(move)
        ply += 1
        legal += 1
        best_score = max(best_score, -negamax(board, depth-1, -beta, -alpha))
        board.pop()
        ply -= 1
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break
    if legal == 0:
        if incheck:
            return -infinity + ply
        else:
            return 0
    return (best_score)

def search_pos(board, depth, alpha, beta):
    global ply
    best_move = None
    best_score = -infinity
    for move in board.legal_moves:
        board.push(move)
        ply += 1
        score = -negamax(board, depth-1, alpha, beta)
        board.pop()
        ply -= 1
        if score > best_score:
            best_score = score
            best_move = move
        if is_time_limit_reached():
            break
    return best_move

def qSearch(board, alpha, beta):
    if is_time_limit_reached():
        return 0
    score = Heuristics.evaluate(board)
    if score >= beta:
        return beta
    if score > alpha:
        alpha = score
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -qSearch(board, -beta, -alpha)
            board.pop()
            if score > alpha:
                if score >= beta:
                    return beta
                alpha = score
    return alpha

def get_pos_val(board, PieceType, table):
        white = 0
        black = 0
        white_pieces = board.pieces(PieceType,chess.WHITE)
        for sq in white_pieces:
            white += table[sq]
            white += get_piece_val(PieceType)
        black_pieces = board.pieces(PieceType,chess.BLACK)
        for sq in black_pieces:
            black += table[sq^56]
            black += get_piece_val(PieceType)
        return white - black

def is_time_limit_reached():
    global start_time
    global timems

    return 1000 * (time.perf_counter() - start_time) >= timems

print(board)
print("moves played: ")
print(movecount)
timems = MAX_TIME_MS

while True:

    move = input("Make a move: \n")
    move = chess.Move.from_uci(move)
    board.push(move)
    print(board)

    start_time = time.perf_counter()
    ply = 0
    ai_move = search_pos(board, 2, -infinity, infinity)
    board.push(ai_move)
    print("Beetle played: ")
    print(ai_move)
    print(board)
    
#    move = input("Make a move: \n")
#    move = chess.Move.from_uci(move)
#    board.push(move)
#    print(board)

    movecount += 1
    print("moves played: ")
    print(movecount)