'''
CPSC 415 -- Homework #3 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import tkinter as tk
import tkinter.font as tkFont
import tkinter.ttk as ttk
import sys
import collections
import builtins
import time
import random

import chess_config
import chess_model
import chess_player

FRAMES = 20

class View():

    def __init__(self, root):
        self.game = chess_model.game
        self.do_replay = False

        # photo_images: the PhotoImage objects, one per *type* of piece, that
        # are loaded from disk.
        self.photo_images = {}

        # displayed_images: the actual individual displayed spites, one per
        # *piece*, that are each positioned on an actual square.
        self.displayed_images = {}

        self.custom_font = tkFont.Font(family="{Times}",size=16)
        self.bigger_font = tkFont.Font(family="{Times}",size=20)
        self.root = root
        self.root.title("Stephen's Crazy Chess")
        self.root.bind('<Return>', self.start_game)
        self.top = ttk.Frame(self.root)
        self.top.pack()
        self.setup()
        self._drag_stuff = {'x': 0, 'y': 0, 'piece': None}
        self.center()

    def center(self):
        self.root.update_idletasks()
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight() + 200
        size = tuple(int(_) for _ in self.root.geometry().split('+')[0].split('x'))
        size = (size[0], size[1]+75)
        x = w/2 - size[0]/2
        y = h/2 - size[1]/2 - 175
        self.root.geometry("%dx%d+%d+%d" % (size + (x, y)))


    def replay(self, saved_game, replay_speed, final_message=""):
        self.saved_game = saved_game
        self.replay_speed = replay_speed
        self.final_message = final_message
        self.game_type_var.set('Computer v Computer')
        self.do_replay = True
        self.start_game()
        self.root.after(10, self._replay_moves, saved_game['MOVES'],
            saved_game['TIMES'][2:] + saved_game['TIMES'][-2:])  # weird

    def _replay_moves(self, moves, times):
        self.root.after(5, self._animate_move, moves[0], times[0])
        if len(moves) > 1:
            self.root.after(self.replay_speed, self._replay_moves, moves[1:],
                times[1:])

    def setup(self):
        self.create_top_frame()
        self.create_canvas()
        self.create_bottom_frame()

    def create_top_frame(self):
        self.status_bar = tk.Frame(self.top)
        self.status_bar.pack(padx=10,pady=10)
        ttk.Label(self.status_bar, text="Turn: ",
            font=self.custom_font).pack(side="left")
        self.turn_label = ttk.Label(self.status_bar,
            text=self.game.player_turn.capitalize(), width=5,
            font=self.custom_font)
        self.turn_label.pack(side="left")
        self.status_label = ttk.Label(self.status_bar, text='',
            foreground='red', anchor=tk.CENTER, width=40,
            font=self.custom_font)
        self.status_label.pack(side="right")
        self.content = ttk.Frame(self.top)
        self.content.pack()

    def create_bottom_frame(self):
        self.player_bar = tk.Frame(self.top)
        self.player_bar.pack(padx=10,pady=10)

        self.board_type_var = tk.StringVar(self.player_bar,
            cfg.config_file_basename.title())
        self.board_menu = tk.OptionMenu(self.player_bar, self.board_type_var,
            *[ c.title() for c in cfg.get_config_names() ])

        self.crazy_mode_var = tk.BooleanVar(value=cfg.CRAZY)
        self.crazy_mode_check = tk.Checkbutton(self.player_bar,
            text="Crazy mode", variable=self.crazy_mode_var)

        self.game_type_var = tk.StringVar(self.player_bar, 'Human v Human')
        self.game_menu = tk.OptionMenu(self.player_bar, self.game_type_var,
            'Human v Human','Human v Computer','Computer v Computer')

        opponent_types = chess_player.ChessPlayer.get_player_names()
        self.opponent1_type_var = tk.StringVar(self.player_bar,
            opponent_types[0])
        self.opponent2_type_var = tk.StringVar(self.player_bar,
            opponent_types[0])
        self.opponent1_menu = tk.OptionMenu(self.player_bar,
           self.opponent1_type_var, *opponent_types)
        self.opponent2_menu = tk.OptionMenu(self.player_bar,
           self.opponent2_type_var, *opponent_types)
        self.opponent1_menu.config(state='disabled')
        self.opponent2_menu.config(state='disabled')

        def game_type_changed(*args):
            new_type = self.game_type_var.get()
            if new_type == 'Human v Human':
                self.opponent1_menu.config(state='disabled')
                self.opponent2_menu.config(state='disabled')
            elif new_type == 'Human v Computer':
                self.opponent1_menu.config(state='normal')
                self.opponent2_menu.config(state='disabled')
            else:  # Computer v Computer
                self.opponent1_menu.config(state='normal')
                self.opponent2_menu.config(state='normal')
        self.game_type_var.trace('w', game_type_changed)

        self.start_game_button = tk.Button(self.player_bar, text='Start game',
            command=self.start_game)

        self.player_bar.rowconfigure(1, weight=1)
        self.player_bar.columnconfigure(3, weight=1)
        self.board_menu.grid(row=0,column=0)
        self.crazy_mode_check.grid(row=1,column=0,columnspan=2)
        self.game_menu.grid(row=0,column=1)
        self.opponent1_menu.grid(row=0,column=2)
        self.opponent2_menu.grid(row=1,column=2)
        self.start_game_button.grid(row=0,column=3)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.content)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind('<ButtonPress-1>', self.handle_drag_start)
        self.canvas.bind('<B1-Motion>', self.handle_drag)
        self.canvas.bind('<ButtonRelease-1>', self.handle_drag_stop)
        self.canvas.bind('<Button-3>', self.handle_square_rt_click)

    def draw_board(self):
        curr_col = cfg.BOARD_COLOR_LIGHT
        for x in range(0,cfg.NUM_COLS*cfg.SQUARE_WIDTH,
                                                        cfg.SQUARE_WIDTH):
            for y in range(0,cfg.NUM_ROWS*cfg.SQUARE_HEIGHT,
                                                        cfg.SQUARE_HEIGHT):
                x2, y2 = x + cfg.SQUARE_WIDTH, y + cfg.SQUARE_HEIGHT
                self.canvas.create_rectangle(x, y, x2, y2, fill=curr_col)
                curr_col = (cfg.BOARD_COLOR_LIGHT
                    if curr_col == cfg.BOARD_COLOR_DARK
                    else cfg.BOARD_COLOR_DARK)
            curr_col = (cfg.BOARD_COLOR_LIGHT
                if curr_col == cfg.BOARD_COLOR_DARK
                else cfg.BOARD_COLOR_DARK)

    def handle_drag_start(self, event):
        loc = self.get_clicked_location(event)
        piece = self.game.board.get(loc,None)
        if self.accepting_input() and piece:
            if piece.color == self.game.player_turn:
                self._drag_stuff['x'], self._drag_stuff['y'] = event.x, event.y
                self._drag_stuff['piece'] = piece
                self.show_hints_for(self.game.board[loc], loc)
                self.canvas.tag_raise(self.displayed_images[piece])
            else:
                self.display_status_message("Not {}'s turn!".format(
                    piece.color))

    def display_status_message(self, message, color='red', delay_ms=None):
        def remove_text():
            self.status_label['text'] = ''
        self.status_label['text'] = message
        self.status_label['foreground'] = color
        if delay_ms:
            self.status_label.after(delay_ms, remove_text)
        self.status_label.update()

    def handle_drag(self, event):
        if self._drag_stuff['piece']:
            delta_x = event.x - self._drag_stuff['x']
            delta_y = event.y - self._drag_stuff['y']
            self.canvas.move(self.displayed_images[self._drag_stuff['piece']],
                delta_x, delta_y)
            self._drag_stuff['x'], self._drag_stuff['y'] = event.x, event.y

    def handle_drag_stop(self, event):
        '''The user has attempted to move a piece.'''
        loc = self.get_clicked_location(event)
        the_piece = self._drag_stuff['piece']
        if not the_piece:
            return
        orig_loc = [ l for l, p in self.game.board.items()
                                                        if p == the_piece ][0]
        self.attempt_to_make_move(the_piece, orig_loc, loc)
        self._drag_stuff = {'x': 0, 'y': 0, 'piece': None}
        self.canvas.after(1, self.canvas.delete, 'hint')

    def attempt_to_make_move(self, the_piece, orig_loc, loc, the_time=None):
        try:
            self.game.board.make_move(orig_loc, loc)

            opp_color = 'black' if the_piece.color == 'white' else 'white'
            if self.game.board.is_king_in_checkmate(opp_color):
                self.display_status_message('CHECKMATE!!!!','red')
                self.end_game()
                return
            if self.game.board._is_stalemated(opp_color):
                self.display_status_message('STALEMATE!','purple')
                self.end_game()
                return
            if len(self.game.board.moves) > cfg.MAX_MOVES * 2:
                self.display_status_message('Draw (too many moves)!','purple')
                self.end_game()
                return
            if self.game.board.is_king_in_check(opp_color):
                self.display_status_message(
                    '{} in check'.format(opp_color.capitalize()),'black')
            else:
                self.display_status_message('')
            self.switch_player_turn(the_time)
        except chess_model.IllegalMoveException as err:
            self.display_status_message(err.args[0])
        self.draw_pieces()

    def end_game(self):
        self.game.started = False
        if self.do_replay:
            [ c.grid_forget() for c in self.player_bar.winfo_children() ]
            final_label = (ttk.Label(self.player_bar, text=self.final_message,
                font=self.bigger_font, foreground='darkgreen', 
                    anchor=tk.CENTER).grid(row=0,column=0,columnspan=3,
                                                                sticky='WE'))
        else:
            self.game.write_log()
        self.draw_pieces()

    def start_game(self, event=None):
        if self.do_replay:
            builtins.cfg = \
                chess_config.Config(self.saved_game['CONFIG'].lower(), True)
            cfg.START_POSITION = self.saved_game['STARTING_POS']
        else:
            builtins.cfg = \
                chess_config.Config(self.board_type_var.get().lower(),
                self.crazy_mode_var.get())
        self.game._reset()
        width = cfg.NUM_COLS * cfg.SQUARE_WIDTH
        height = cfg.NUM_ROWS * cfg.SQUARE_HEIGHT
        self.canvas.config(width=width, height=height)
        self.draw_board()
        self.draw_pieces()
        if self.do_replay:
            self.game.white_player = self.saved_game['WHITE']
            self.game.black_player = self.saved_game['BLACK']
            game_type_text = (self.game.white_player + ' versus ' +
                self.game.black_player)
        elif self.game_type_var.get() == 'Human v Human':
            game_type_text = 'Human versus Human'
            self.game.white_player = 'Human'
            self.game.black_player = 'Human'
        elif self.game_type_var.get() == 'Human v Computer':
            bot_module = __import__(self.opponent1_type_var.get())
            opponent_class = getattr(bot_module,
                self.opponent1_type_var.get())
            self.black_opponent = opponent_class(self.game.board, 'black')
            b_name = self.opponent1_type_var.get().replace('_ChessPlayer','')
            game_type_text = 'Human versus ' + b_name
            self.game.white_player = 'Human'
            self.game.black_player = b_name
        else:  # Computer v Computer
            bot1_module = __import__(self.opponent1_type_var.get())
            white_opponent_class = getattr(bot1_module,
                self.opponent1_type_var.get())
            self.white_opponent = white_opponent_class(self.game.board,'white')
            bot2_module = __import__(self.opponent2_type_var.get())
            black_opponent_class = getattr(bot2_module,
                self.opponent2_type_var.get())
            self.black_opponent = black_opponent_class(self.game.board,'black')
            w_name = self.opponent1_type_var.get().replace('_ChessPlayer','')
            b_name = self.opponent2_type_var.get().replace('_ChessPlayer','')
            game_type_text = w_name + ' versus ' + b_name
            self.game.white_player = w_name
            self.game.black_player = b_name
        [ c.grid_forget() for c in self.player_bar.winfo_children() ]
        ttk.Label(self.player_bar, text=game_type_text,
            font=self.custom_font, anchor=tk.CENTER).grid(
                row=0,column=0,columnspan=3,sticky='WE')
        self.progress_value = tk.DoubleVar()
        self.progress_value.set(0.)
        self.progress_bar = ttk.Progressbar(self.player_bar,
            mode='determinate', variable=self.progress_value,
            orient=tk.HORIZONTAL)
        self.progress_bar.value = self.progress_value
        self.progress_bar.grid(row=1,columnspan=3, sticky='WE')
        self.progress_text = tk.StringVar()
        ttk.Label(self.player_bar, textvariable=self.progress_text,
            font=self.custom_font, anchor=tk.CENTER).grid(
                row=2,column=0,columnspan=3,sticky='WE')
        self.progress_stuff = collections.namedtuple('ProgressStuff',
            ['bar','text'])(self.progress_bar, self.progress_text)
        self.player_bar.rowconfigure(2, weight=1)
        self.player_bar.columnconfigure(1, weight=1)
        self.player_time = {'white':0.0,'black':0.0}
        self.player_time_label = {}
        self.player_time_label['white'] = ttk.Label(self.player_bar, 
            text='0.0 sec', font=self.custom_font, anchor=tk.CENTER, 
                foreground='black', background='white')
        self.player_time_label['white'].grid(row=3,column=0,sticky='W')
        ttk.Label(self.player_bar, text='',
            font=self.custom_font, anchor=tk.CENTER, foreground='black').grid(
                row=3,column=1,sticky='WE')
        self.player_time_label['black'] = ttk.Label(self.player_bar, 
            text='0.0 sec', font=self.custom_font, anchor=tk.CENTER, 
                foreground='white', background='black')
        self.player_time_label['black'].grid(row=3,column=2,sticky='E')
        self.turn_label['text'] = 'White'
        self.timer = time.perf_counter()
        self.center()
        if not self.do_replay:
            self.take_player_turn()

    def switch_player_turn(self, the_time):
        if self.do_replay:
            self.player_time[self.game.player_turn] = the_time
        else:
            time_against = time.perf_counter() - self.timer
            self.player_time[self.game.player_turn] += time_against
        self.player_time_label[self.game.player_turn]['text'] = \
            '{:.1f} sec'.format(self.player_time[self.game.player_turn])
        if self.player_time[self.game.player_turn] > cfg.TIME_LIMIT:
            self.player_time_label[self.game.player_turn]['foreground'] = 'red'

        self.turn_label['foreground'] = self.game.player_turn
        self.game.player_turn = \
            'black' if self.game.player_turn == 'white' else 'white'
        self.turn_label['text'] = self.game.player_turn.capitalize()
        self.turn_label['background'] = self.game.player_turn
        if not self.do_replay:
            if self.player_time[self.game.player_turn] > cfg.TIME_LIMIT:
                self.force_random_move()
            else:
                self.take_player_turn()

    def _animate_move(self, move, the_time=None):
        the_image = self.displayed_images[self.game.board[move[0]]]
        orig_x, orig_y = self.calculate_piece_coords(move[0])
        new_x, new_y = self.calculate_piece_coords(move[1])
        delta_x = int((new_x - orig_x) / FRAMES)
        delta_y = int((new_y - orig_y) / FRAMES)
        def moveme(delta_x, delta_y):
            self.canvas.move(the_image, delta_x, delta_y)
            self.canvas.update_idletasks()
        for i in range(FRAMES+1):
            self.canvas.after(i*10, moveme, delta_x, delta_y)
        self.canvas.after((FRAMES+5)*10,self.attempt_to_make_move,
            self.game.board[move[0]], *move, the_time)

    def take_player_turn(self):
        self.timer = time.perf_counter()
        if self.accepting_input():
            self.root.config(cursor='')
        else:
            self.root.config(cursor='watch')
            self.root.update()
            the_moving_opponent = \
                self.white_opponent if self.game.player_turn == 'white' \
                else self.black_opponent
            other_player = 'white' if self.game.player_turn == 'black' \
                else 'black'
            move = the_moving_opponent.get_move(
                cfg.TIME_LIMIT - self.player_time[self.game.player_turn],
                cfg.TIME_LIMIT - self.player_time[other_player],
                self.progress_stuff)
            self.root.after(1, self._animate_move, move)

    def force_random_move(self):
        self.timer = time.perf_counter()
        random_move = random.choice(
            self.game.board.get_all_available_legal_moves(
                self.game.player_turn))
        self.display_status_message('{} out of time! Forced random move.'.
            format(self.game.player_turn.capitalize()),'red')
        self.root.after(0, self._animate_move, random_move)

    def handle_square_rt_click(self, event):
        loc = self.get_clicked_location(event)
        self.show_hints_for(self.game.board[loc], loc)

    def show_hints_for(self, piece, loc):
        if piece:
            for m in [ 
                l for _,l in piece._moves_available(loc) ]:
                try:
                    self.game.board._assert_legal_move(loc, m)
                    center = self.calculate_piece_coords(m)
                    self.canvas.create_rectangle(
                        center[0] - cfg.SQUARE_WIDTH //2,
                        center[1] - cfg.SQUARE_HEIGHT //2,
                        center[0] + cfg.SQUARE_WIDTH //2,
                        center[1] + cfg.SQUARE_HEIGHT //2,
                        fill='red', tags='hint')
                    if self.game.board.get(m, None):
                        self.canvas.lift(
                            self.displayed_images[self.game.board[m]])
                        self.canvas.update_idletasks()
                except chess_model.IllegalMoveException as e:
                    pass
            self.canvas.after(3000, self.canvas.delete, 'hint')

    def get_clicked_location(self, event):
        clicked_x = cfg.X_AXIS_LABELS[event.x // cfg.SQUARE_WIDTH]
        clicked_y = cfg.Y_AXIS_LABELS[
            cfg.NUM_ROWS - 1 - (event.y // cfg.SQUARE_HEIGHT)]
        return clicked_x + clicked_y

    def draw_piece(self, piece, location):
        filename = '{}/{}'.format(cfg.IMAGE_DIR,piece._get_filename())
        if filename not in self.photo_images:
            self.photo_images[filename] = tk.PhotoImage(file=filename)
        self.displayed_images[piece] = self.canvas.create_image(
            *self.calculate_piece_coords(location),
            image=self.photo_images[filename], tags='piece')

    def calculate_piece_coords(self, location):
        x_coord = cfg.X_AXIS_LABELS.index(location[0])
        y_coord = cfg.NUM_ROWS - 1 - cfg.Y_AXIS_LABELS.index(location[1])
        return (x_coord * cfg.SQUARE_WIDTH + cfg.SQUARE_WIDTH // 2,
                y_coord * cfg.SQUARE_HEIGHT + cfg.SQUARE_HEIGHT // 2)

    def draw_pieces(self):
        self.canvas.delete('piece')
        for location, piece in self.game.board.items():
            self.draw_piece(piece, location)

    def accepting_input(self):
        return (self.game.started and (
            self.game_type_var.get() == 'Human v Human' or
            (self.game_type_var.get() == 'Human v Computer' and
                self.game.player_turn == 'white')))
