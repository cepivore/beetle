
import time

import chess
import chess.pgn


INF = 100000
MATE_SCORE = INF-100
MAX_DEPTH = 16
MAX_TIME_MS = 120000  # 120s or 2m


# Evaluating the board
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]


def is_time_limit_reached():
    global start_time
    global timems

    return 1000 * (time.perf_counter() - start_time) >= timems


def evaluate_board(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq

    if board.turn:
        return eval
    return -eval


def value_to_mate(value):
    if (value < -MATE_SCORE):
        return -(INF + value) // 2
    
    elif (value > MATE_SCORE):
        return (INF - value + 1) // 2

    return 0


# Searching the best move using minimax and alphabeta algorithm with negamax implementation
def alphabeta(board, alpha, beta, depth):
    global timems
    global start_time
    global ply

    bestscore = -INF

    if is_time_limit_reached():
        return 0

    if board.can_claim_draw() or board.is_insufficient_material():
        return 0

    incheck = board.is_check()
    if incheck:
        depth += 1

    if (depth < 1):
        return quiesce(board, alpha, beta)

    legal = 0
    for move in board.legal_moves:
        board.push(move)
        ply += 1
        legal += 1

        score = -alphabeta(board, -beta, -alpha, depth - 1)
        board.pop()
        ply -= 1

        if (score >= beta):
            return score

        if (score > bestscore):
            bestscore = score

        if (score > alpha):
            alpha = score

    if legal == 0:
        if incheck:
            return -INF + ply
        else:
            return 0

    return bestscore


def quiesce(board, alpha, beta):
    if is_time_limit_reached():
        return 0

    if board.is_insufficient_material():
        return 0

    stand_pat = evaluate_board(board)
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)

            score = -quiesce(board, -beta, -alpha)
            board.pop()

            if (score >= beta):
                return beta

            if (score > alpha):
                alpha = score

    return alpha


def selectmove(board, depth):
    global ply 

    bestMove = chess.Move.null()
    bestValue = -INF
    alpha = -INF
    beta = INF

    for move in board.legal_moves:
        board.push(move)
        ply += 1

        boardValue = -alphabeta(board, -beta, -alpha, depth - 1)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move

        if (boardValue > alpha):
            alpha = boardValue

        board.pop()
        ply -= 1

        if is_time_limit_reached():
            break

    return bestMove, bestValue


if __name__ == '__main__':
    global timems
    global start_time
    global ply

    fen = chess.STARTING_FEN
    # fen = "1k6/7Q/1K6/8/8/8/8/8 w - - 0 1"  # mate in 1
    # fen = "kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1"  # mate in 2
    # fen = "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"  # sicilian
    
    depth = MAX_DEPTH
    timems = MAX_TIME_MS

    board = chess.Board(fen)

    while True:
        userinput = input('> ')
        line = userinput.rstrip()

        if line == 'go':
            bestmove, bestscore = None, None
            start_time = time.perf_counter()
            ply = 0

            if not board.is_game_over():
                for d in range(1, depth+1):
                    move, score = selectmove(board, d)

                    elapse = 1000 * (time.perf_counter() - start_time)

                    if elapse >= timems:
                        break  # Don't overwrite our bestmove from the time-interrupted search.

                    bestmove = move
                    bestscore = score

                    if abs(bestscore) > MATE_SCORE:
                        mate = value_to_mate(bestscore)
                        print(f'info depth {d} score mate {mate} time {int(elapse)} pv {bestmove}')  # time is in millisec
                    else:
                        print(f'info depth {d} score {bestscore} time {int(elapse)} pv {bestmove}')  # time is in millisec

                board.push(bestmove)  # update board with the bestmove
                print(f'info time {int(1000*(time.perf_counter() - start_time))}')
                print(f'bestmove {bestmove}')
            else:
                print('Game is over!')

        elif line == 'new':
            board = chess.Board()
            depth = MAX_DEPTH
            timems = MAX_TIME_MS
            print(board)
            print(f'max depth: {depth}, max time ms: {timems}')

        elif 'position fen ' in line:
            fen = ' '.join(line.split()[2:7])
            board = chess.Board(fen)

        elif line == 'board':
            print(board)

        elif line == 'fen':
            print(board.fen())

        elif line == 'epd':
            print(board.epd())

        elif 'time' in line:
            timems = int(line.split()[1])

        elif 'depth' in line:
            depth = int(line.split()[1])

        elif line == 'game':
            game = chess.pgn.Game()
            print(game.from_board(board))

        elif line == 'quit':
            break

        else:
            # assume user has entered a move
            try:
                board.push(chess.Move.from_uci(line))
            except ValueError:
                print(f'illegal move {line}')
