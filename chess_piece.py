'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

from abc import ABC


'''Piece objects represent individual pieces on the board. The only thing you
should have to do to a Piece object is call .get_notation() to get the chess
notation for its piece type (e.g., "R" for a white rook, or "n" for a black
knight.

Anything else is considered cheating, and will result in public flogging.
'''
class Piece(ABC):
    @classmethod
    def from_notation(cls, notation, board):
        color = 'white' if notation.isupper() else 'black'
        if notation.upper() in cfg.SHORT_NAME:
            piece_classname = cfg.SHORT_NAME[notation.upper()]
            return globals()[piece_classname](color, board)
        else:
            raise Exception('No such piece "{}"'.format(notation))

    def __init__(self, color, board):
        self.board = board
        self.color = color
        self.name = self.__class__.__name__.lower()

    def get_notation(self):
        piece_name = cfg.PIECE_NAME[self.name.capitalize()]
        return piece_name if self.color == 'white' else piece_name.lower()

    def __str__(self):
        return 'a {} {}'.format(self.color, self.name)

    def _get_filename(self):
        return '{}_{}.gif'.format(cfg.PIECE_NAME[self.name.capitalize()],
            self.color.lower())

    def _moves_available(self, current_loc):
        allowed_moves = []
        for x,y in self.directions:
            collision = False
            step = 1
            while not collision and step <= self.max_dist:
                destination = (chr(ord(current_loc[0]) + y * step) +
                    str(int(current_loc[1]) + x * step))
                if destination not in self.board.all_occupied_positions():
                    # Square clear. Can go here and past.
                    allowed_moves.append(destination)
                elif self.board[destination].color == self.color:
                    # Ran into same color piece as me. Can't go here or past.
                    collision = True
                else:
                    # Ran into opposite color piece. Can go here, but not
                    # past.
                    allowed_moves.append(destination)
                    collision = True
                step += 1
        allowed_moves = [ (current_loc, move) for move in allowed_moves
            if move[0] in cfg.X_AXIS_LABELS and
            move[1:] in cfg.Y_AXIS_LABELS ]
        return allowed_moves

    def _move_yourself(self, orig_loc, loc):
        # Default is just to move myself from here to there.
        self.board[loc] = self.board.pop(orig_loc)


class King(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.ORTHOGONAL_DIRS + cfg.DIAGONAL_DIRS
        self.has_moved = False
        self.max_dist = 1
        self.orig_square = [ l for l,p in cfg.START_POSITION.items()
            if p == self.get_notation() ][0]

    def _moves_available(self, current_loc):
        moves = super()._moves_available(current_loc)
        if not self.has_moved and self._get_castling_rook('king'):
            moves.append((current_loc, right_from(current_loc, 2)))
        if not self.has_moved and self._get_castling_rook('queen'):
            moves.append((current_loc, left_from(current_loc, 2)))
        return moves

    def _get_castling_rook(self, side='king'):
        '''If there is a legit castling rook on the requested side, return a
        tuple containing it and its location. Otherwise, return None.
        "Legit" means both (1) the rook is on that square, and (2) it hasn't
        been moved, and (3) there are no pieces between it and the king.'''
        rook_piece_name = 'r' if self.color == 'black' else 'R'
        rook_locs = [ l for l,p in cfg.START_POSITION.items()
                if p == rook_piece_name and
                (side == 'king' and l[0] == cfg.X_AXIS_LABELS[-1] or
                 side == 'queen' and l[0] == cfg.X_AXIS_LABELS[0])]
        if len(rook_locs) < 1:
            return None
        rook_loc = rook_locs[0]
        rook = self.board.get(rook_loc, None)
        if rook and isinstance(rook, Rook) and not rook.has_moved:
            direction = -1 if side == 'king' else 1
            intermediate_pos = right_from(rook_loc, direction)
            while intermediate_pos != self.orig_square:
                if intermediate_pos in self.board.keys():
                    return None
                intermediate_pos = right_from(intermediate_pos, direction)
            return (rook, rook_loc)
        return None

    def _move_yourself(self, orig_loc, loc):
        self.has_moved = True
        if (orig_loc == self.orig_square and
                        loc == right_from(self.orig_square, 2)):
            # Castling king-side.
            rook = self._get_castling_rook('king')
            rook[0]._move_yourself(rook[1],       # Move rook.
                right_from(self.orig_square,1))
        elif (orig_loc == self.orig_square and
                        loc == left_from(self.orig_square, 2)):
            # Castling queen-side.
            rook = self._get_castling_rook('queen')
            rook[0]._move_yourself(rook[1],       # Move rook.
                left_from(self.orig_square,1))
        super()._move_yourself(orig_loc, loc)     # Move king.


class Queen(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.ORTHOGONAL_DIRS + cfg.DIAGONAL_DIRS
        self.max_dist = max(cfg.NUM_ROWS, cfg.NUM_COLS)


class Princess(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.ORTHOGONAL_DIRS + cfg.DIAGONAL_DIRS
        self.max_dist = 3


class Fool(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.FOOL_DIRS
        self.max_dist = 1


class Rook(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.ORTHOGONAL_DIRS
        self.max_dist = max(cfg.NUM_ROWS, cfg.NUM_COLS)
        self.has_moved = False

    def _move_yourself(self, orig_loc, loc):
        super()._move_yourself(orig_loc, loc)
        self.has_moved = True


class Bishop(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)
        self.directions = cfg.DIAGONAL_DIRS
        self.max_dist = max(cfg.NUM_ROWS, cfg.NUM_COLS)


class Pawn(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)

    def _moves_available(self, current_loc):
        allowed_moves = []

        f = 1 if self.color == 'white' else -1

        forward = current_loc[0] + str(int(current_loc[1]) + f)
        if forward not in self.board.all_occupied_positions():
            allowed_moves.append(forward)

        fwd_left = chr(ord(current_loc[0]) - f) + str(int(current_loc[1]) + f)
        fwd_rt = chr(ord(current_loc[0]) + f) + str(int(current_loc[1]) + f)
        for move in [ fwd_left, fwd_rt ]:
            if move in self.board.all_occupied_positions(
                    'black' if self.color == 'white' else 'white'):
                allowed_moves.append(move)

        fwd_two = current_loc[0] + str(int(current_loc[1]) + 2*f)
        if ((int(current_loc[1]) == 2 and self.color == 'white' or
                int(current_loc[1]) == cfg.NUM_ROWS - 1 and self.color == 'black') and
                fwd_two not in self.board.all_occupied_positions() and
                forward not in self.board.all_occupied_positions()):
            allowed_moves.append(fwd_two)

        allowed_moves = [ (current_loc, move) for move in allowed_moves
            if move[0] in cfg.X_AXIS_LABELS and
            move[1:] in cfg.Y_AXIS_LABELS ]
        return allowed_moves

    def _move_yourself(self, orig_loc, loc):
        if int(loc[1]) in [1,cfg.NUM_ROWS]:
            # Pawn promotion.
            new_queen = Queen(self.color, self.board)
            del self.board[orig_loc]
            self.board[loc] = new_queen
        else:
            super()._move_yourself(orig_loc, loc)


class Knight(Piece):
    def __init__(self, color, board):
        super().__init__(color, board)

    def _moves_available(self, current_loc):
        allowed_moves = []
        for x,y in cfg.KNIGHT_DIRS:
            destination = (chr(ord(current_loc[0]) + y) +
                (str(int(current_loc[1]) + x)))
            if destination not in self.board.all_occupied_positions(
                    self.color):
                allowed_moves.append(destination)
        allowed_moves = [ (current_loc, move) for move in allowed_moves
            if move[0] in cfg.X_AXIS_LABELS and
            move[1:] in cfg.Y_AXIS_LABELS ]
        return allowed_moves


def left_from(loc, steps=1):
    return chr(ord(loc[0]) - steps) + loc[1]

def right_from(loc, steps=1):
    return left_from(loc, -steps)

