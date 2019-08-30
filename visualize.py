'''
CPSC 415
Stephen Davies, University of Mary Washington, fall 2019
'''

import tkinter as tk
import tkinter.ttk as ttk
import logging

from environment import *

class VisualXYEnvironment:
    '''A "view" to the XYEnvironment's "model".'''

    IMAGE_DIR = 'images'
    SMALL = False

    def __init__(self, xy_env, max_thing_width, max_thing_height,
            title='CPSC 415'):
        '''Initialize this view with a model to give a view of.'''
        self.xy_env = xy_env
        if self.SMALL:
            max_thing_width /= 2
            max_thing_height /= 2
        self.CANVAS_WIDTH = max_thing_width * self.xy_env.width
        self.CANVAS_HEIGHT = max_thing_height * self.xy_env.height
        self.square_width = max_thing_width
        self.square_height = max_thing_height
        self.title = title
        self.image_cache = {}
        self._setup_graphics()
        self.xy_env.add_observer(self)
        self.still_running = True
        self.total_steps = 0

    def start(self, interactive):
        logging.info('starting {}...'.format(
            'interactive' if interactive else 'auto'))
        self.draw_entire_environment()
        if not interactive:
            self.continuous.set(True)
            self.delay.set(1)
            self.run_until(10000000000)
        self.root_window.mainloop()
        logging.info('...done!')

    def thing_moved(self, thing, animation_start_end=()):
        '''If animation_start_end is empty, simply record that this thing has
        been moved. Otherwise, display it moving from one location to the
        other (the elements of the tuple).'''
        if animation_start_end:
            self.animate_thing(thing, *animation_start_end)
        else:
            self.redraw_entire_environment()

    def thing_deleted(self, thing):
        self.redraw_entire_environment()

    def calculate_coords(self, loc):
        return (self.square_width * loc[0],
                self.CANVAS_HEIGHT - self.square_height * (loc[1] + 1))

    def animate_thing(self, thing, loc1, loc2):
        def do_anim():
            FRAMES = 20
            the_image = self.draw(thing, loc1)
            orig_x, orig_y = self.calculate_coords(loc1)
            new_x, new_y = self.calculate_coords(loc2)
            delta_x = int((new_x - orig_x) / FRAMES)
            delta_y = int((new_y - orig_y) / FRAMES)
            def moveme(delta_x, delta_y):
                self.canvas.move(the_image, delta_x, delta_y)
                self.canvas.update_idletasks()
            for i in range(FRAMES+1):
                self.canvas.after(i*10, moveme, delta_x, delta_y)
            self.canvas.after((FRAMES+5)*10,lambda:
                self.canvas.delete(the_image))
        self.canvas.after(1, do_anim)


    def draw_entire_environment(self):
        self.canvas.delete('thing')
        for thing, loc in self.xy_env.items():
            self.draw(thing, loc)

    def redraw_entire_environment(self):
        self.draw_entire_environment()

    def draw(self, thing, loc):
        x, y = self.calculate_coords(loc)
        if self.SMALL:
            basename = thing.image_filename.replace('.','_small.')
        else:
            basename = thing.image_filename
        filename = '{}/{}'.format(self.IMAGE_DIR,basename)
        if filename not in self.image_cache:
            self.image_cache[filename] = tk.PhotoImage(file=filename)
        return self.canvas.create_image(x + self.square_width // 2,
            y + self.square_height // 2,
            image=self.image_cache[filename], tags='thing')

    def run_until(self, steps=1000):
        self.progress.set("Vroom! ({} steps)".format(self.total_steps))
        if self.total_steps > steps:
            self.still_running = False
        self.total_steps += 1
        if self.still_running:
            self.xy_env.step()
            self.score.set(str(self.xy_env.agents[0].performance
                    if self.xy_env.agents else '') + ' pts')
            if self.continuous.get() and not self.xy_env.should_shutdown():
                self.root_window.after(int(self.delay.get()),
                    self.run_until,steps)
            elif self.continuous.get() and self.xy_env.should_shutdown():
                print('Finished in {} moves for a score of {}!'
                    .format(self.total_steps, self.score.get()))

    def _setup_graphics(self):
        self.root_window = tk.Tk()
        self.root_window.grid()
        self.root_window.title(self.title)
        self.root_window.config(background='white')

        self.progress = tk.StringVar() ; self.progress.set('')
        ttk.Label(self.root_window,background='white',foreground='red',
            anchor='center', textvariable=self.progress).grid(
            row=2,column=0,columnspan=4)

        self.score = tk.StringVar() ; self.score.set('')
        ttk.Label(self.root_window,background='white',foreground='blue',
            anchor='center', textvariable=self.score).grid(
            row=2,column=4,columnspan=1)

        ttk.Label(self.root_window,text='# iterations:',
            background='white').grid(row=1,column=0, sticky='e')

        num_iter_var = tk.IntVar() ; num_iter_var.set(200)

        ttk.Entry(self.root_window,textvariable=num_iter_var,width=5).grid(
            row=1,column=1,sticky='w')
        ttk.Button(self.root_window,text='Go',
            command=lambda : self.run_until(num_iter_var.get())).grid(
            row=1,column=2, sticky='W')
        self.continuous = tk.BooleanVar()
        ttk.Checkbutton(self.root_window,text='Continuous',
            variable=self.continuous).grid(row=1,column=3)

        ttk.Label(self.root_window,text='delay (ms):',
            background='white').grid(row=1,column=4, sticky='e')
        self.delay = tk.StringVar()
        tk.Spinbox(self.root_window,values=[10,50,100,500,1000],width=4,
            textvariable=self.delay).grid(row=1,column=5)
        self.delay.set(100)

        self.root_window.bind('<Return>',
            lambda x: self.run_until(num_iter_var.get()))
        self.canvas = tk.Canvas(self.root_window,
            width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg='black')

        self.canvas.grid(row=0,column=0,columnspan=6,sticky='we')

