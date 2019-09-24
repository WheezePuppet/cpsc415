'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import sys
import collections
import builtins
import time
import random

import chess_config
import chess_model
import chess_player

RECORD_TIMES = False

class NonExistentValue():
    def set(self, n):
        pass
    def get(self):
        return 0

class NonExistentText():
    def __init__(self):
        pass
    def set(self, some_string):
        pass

class NonExistentProgressBar():
    def __init__(self):
        self.value = NonExistentValue()
    def config(self, maximum):
        pass
    def update_idletasks(self):
        pass

class NonExistentProgressStuff():
    def __init__(self):
        self.bar = NonExistentProgressBar()
        self.text = NonExistentText()
    def config(self, maximum):
        pass
    def update_idletasks(self):
        pass


class HeadlessGame():

    def __init__(self, agent1, agent2, config_name, crazy_mode=False,
            log_file_suffix=""):
        '''agent1 and agent2 are the "bare" UMW Ids (without "_ChessPlayer"
            at the end.) config_name is the capitalized name of the config,
            like "Mini" or "Reg".'''
        self.move_num = 1
        self.game = chess_model.game
        self.agent1 = agent1
        self.agent2 = agent2
        self.config_name = config_name
        self.crazy_mode = crazy_mode
        self.log_file_suffix = log_file_suffix

    def display_status_message(self, message):
        print(message)

    def start_game(self):
        builtins.cfg = \
            chess_config.Config(self.config_name.lower(), self.crazy_mode)
        self.game._reset()
        agent1_module = __import__(self.agent1 + "_ChessPlayer")
        white_opponent_class = getattr(agent1_module,
            self.agent1 + "_ChessPlayer")
        self.white_opponent = white_opponent_class(self.game.board,'white')
        agent2_module = __import__(self.agent2 + "_ChessPlayer")
        black_opponent_class = getattr(agent2_module,
            self.agent2 + "_ChessPlayer")
        self.black_opponent = black_opponent_class(self.game.board,'black')
        self.game.white_player = self.agent1
        self.game.black_player = self.agent2
        self.player_time = {'white':0.0,'black':0.0}
        self.timer = time.perf_counter()
        move = self.take_player_turn()
        self.attempt_to_make_move(self.game.board[move[0]], *move)
        while self.game.started:
            move = self.switch_player_turn()
            self.attempt_to_make_move(self.game.board[move[0]], *move)

    def end_game(self):
        self.game.started = False
        self.game.write_log(self.log_file_suffix)

    def attempt_to_make_move(self, the_piece, orig_loc, loc):
        try:
            self.game.board.make_move(orig_loc, loc,
                self.player_time[self.game.player_turn])
            opp_color = 'black' if the_piece.color == 'white' else 'white'
            self.game.chess_record += self._record_move(the_piece.color, 
                orig_loc, loc, self.player_time[self.game.player_turn],
                self.game.board.is_king_in_check(opp_color),
                self.game.board.is_king_in_checkmate(opp_color),
                self.game.board._is_stalemated(opp_color),
                len(self.game.board.moves) > cfg.MAX_MOVES * 2)
            if (self.game.board.is_king_in_checkmate(opp_color) or
                self.game.board._is_stalemated(opp_color) or
                len(self.game.board.moves) > cfg.MAX_MOVES * 2):
                self.end_game()
                return
            if the_piece.color == 'black':
                self.move_num += 1
        except chess_model.IllegalMoveException as err:
            self.display_status_message(err.args[0])


    def _record_move(self, color, orig_loc, loc, the_time, check=False,
            checkmate=False, stalemate=False, draw=False):
        score = ""
        notation = " "
        if color == 'white':
            if checkmate:
                score = " 1-0\n"
                self.game.winner = self.game.white_player
                notation = "#"
            elif check:
                notation = "+"
            elif stalemate:
                score = " ½-½\n"
                notation = " Stalemate "
            elif draw:
                score = " ½-½\n"
                notation = " Moves exceeded "
            if RECORD_TIMES:
                the_record = ("{move_num:3}. {orig_loc}-{new_loc}{notation}" +
                    "({the_time:5.1f}){score}").format(
                    move_num=self.move_num, orig_loc=orig_loc, new_loc=loc,
                    the_time=the_time, score=score, notation=notation)
            else:
                the_record = ("{move_num:3}. {orig_loc}-{new_loc}{notation}" +
                    "{score}").format(
                    move_num=self.move_num, orig_loc=orig_loc, new_loc=loc,
                    score=score, notation=notation)
        else:
            if checkmate:
                score = " 0-1"
                self.game.winner = self.game.black_player
                notation = "#"
            elif check:
                notation = "+"
            elif stalemate:
                score = " ½-½"
                notation = " Stalemate "
            elif draw:
                score = " ½-½"
                notation = " Moves exceeded "
            if RECORD_TIMES:
                the_record = (" {orig_loc}-{new_loc}{notation}" +
                    "({the_time:5.1f}){score}\n").format(
                    move_num=self.move_num, orig_loc=orig_loc, new_loc=loc,
                    the_time=the_time, score=score, notation=notation)
            else:
                the_record = (" {orig_loc}-{new_loc}{notation}" +
                    "{score}\n").format(
                    move_num=self.move_num, orig_loc=orig_loc, new_loc=loc,
                    score=score, notation=notation)
        print(the_record, end='')
        return the_record

    def switch_player_turn(self):
        time_against = time.perf_counter() - self.timer
        self.player_time[self.game.player_turn] += time_against
        if self.player_time[self.game.player_turn] > cfg.TIME_LIMIT:
            print("Outta time!!")

        self.game.player_turn = \
            'black' if self.game.player_turn == 'white' else 'white'
        if self.player_time[self.game.player_turn] > cfg.TIME_LIMIT:
            return self.force_random_move()
        else:
            return self.take_player_turn()

    def take_player_turn(self):
        self.timer = time.perf_counter()
        the_moving_opponent = \
            self.white_opponent if self.game.player_turn == 'white' \
            else self.black_opponent
        other_player = 'white' if self.game.player_turn == 'black' \
            else 'black'
        move = the_moving_opponent.get_move(
            cfg.TIME_LIMIT - self.player_time[self.game.player_turn],
            cfg.TIME_LIMIT - self.player_time[other_player],
            NonExistentProgressStuff())
        return move

    def force_random_move(self):
        self.timer = time.perf_counter()
        random_move = random.choice(
            self.game.board.get_all_available_legal_moves(
                self.game.player_turn))
        print('{} out of time! Forced random move.'.
            format(self.game.player_turn.capitalize()))
        return random_move

