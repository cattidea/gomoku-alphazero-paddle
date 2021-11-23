import threading
import queue
from tkinter import *
from tkinter import messagebox

import numpy as np

from config import *

CW = 30
R = 10


class Scaler():
    """ 坐标变换器
    方便对坐标进行放缩
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def bind(self, start, end):
        self.new_start = start
        self.new_end = end
        return self

    def inverse(self):
        return Scaler(self.new_start, self.new_end).bind(self.start, self.end)

    def __call__(self, value):
        length = self.end - self.start
        new_length = self.new_end - self.new_start
        new_value = (value - self.start) / length * new_length + self.new_start
        return new_value


class BinaryScaler():
    """ 二元坐标变换器 """

    def __init__(self, start_x, start_y, end_x, end_y):
        self.X = Scaler(start_x, end_x)
        self.Y = Scaler(start_y, end_y)

    def bind(self, start_x, start_y, end_x, end_y):
        self.X.bind(start_x, end_x)
        self.Y.bind(start_y, end_y)
        return self

    def inverse(self):
        bs = BinaryScaler(self.X.new_start, self.Y.new_start,
                          self.X.new_end, self.Y.new_end)
        bs.bind(self.X.start, self.Y.start, self.X.end, self.Y.end)
        return bs

    def __call__(self, value_x, value_y):
        return self.X(value_x), self.Y(value_y)


class UI():
    """ UI 基类 """

    def __init__(self):
        pass

    def render(self, board, last_move):
        raise NotImplementedError

    def message(self, message):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def input(self):
        raise NotImplementedError

    def game_start(self, game_loop):
        game_loop()


class GUI(UI):
    """ 棋盘图形 UI """
    name = 'GUI'
    POINT_QUEUE = queue.Queue()

    def __init__(self):
        super().__init__()
        self.tk = Tk()
        self.tk.geometry("{}x{}".format(WIDTH*CW+100, HEIGHT*CW+100))
        self.tk.title('Gomoku')
        self.bc = None
        self.canvas = None
        self.figures = {'chess': [], 'flag': [], 'board': []}
        self._init_canvas()
        self._init_board()
        self.last_move = ()

    def _init_canvas(self):
        canvas_width, canvas_height = WIDTH*CW, HEIGHT*CW
        bc = BinaryScaler(0, 0, canvas_width,
                          canvas_height).bind(-1, HEIGHT, WIDTH, -1)
        bc_ = bc.inverse()
        self.bc = bc
        self.canvas_width, self.canvas_height = canvas_width, canvas_height

        canvas = Canvas(self.tk, width=canvas_width,
                        height=canvas_height, bg='orange')
        self.canvas = canvas

        def on_click(event):
            x, y = bc(event.x, event.y)
            x, y = round(x), round(y)
            GUI.POINT_QUEUE.put((x, y))

        canvas.bind("<ButtonRelease-1>", on_click)

    def _line(self, x1, y1, x2, y2, color='black', name='board'):
        bc_ = self.bc.inverse()
        x1, y1 = bc_(x1, y1)
        x2, y2 = bc_(x2, y2)
        figure_id = self.canvas.create_line(x1, y1, x2, y2, fill=color)
        self.figures[name].append(figure_id)

    def _init_board(self):
        bc_ = self.bc.inverse()
        canvas = self.canvas
        for i in range(HEIGHT):
            self._line(0, i, WIDTH-1, i, name='board')
        for i in range(WIDTH):
            self._line(i, 0, i, HEIGHT-1, name='board')
        canvas.place(x=50, y=50, anchor='nw')

    def _circle(self, x, y, radius=R, color='blue', name='chess'):
        bc_ = self.bc.inverse()
        x_pix, y_pix = bc_(x, y)
        figure_id = self.canvas.create_oval(x_pix-radius, y_pix -
                                            radius, x_pix+radius, y_pix+radius, fill=color)
        self.figures[name].append(figure_id)

    def _delete(self, name):
        for figure_id in self.figures[name]:
            self.canvas.delete(figure_id)
        self.figures[name].clear()

    def _clear(self):
        for name in self.figures:
            self._delete(name)

    def render(self, board, last_move):
        self._delete(name='flag')
        self._circle(*last_move, color=COLOR[board[last_move]], name='chess')
        self._line(last_move[0], last_move[1]-0.2, last_move[0], last_move[1]+0.2,
                   color=COLOR[-board[last_move]], name='flag')
        self._line(last_move[0]-0.2, last_move[1], last_move[0]+0.2, last_move[1],
                   color=COLOR[-board[last_move]], name='flag')

    def message(self, message):
        messagebox.showinfo('INFO', message)

    def reset(self):
        self._clear()
        self._init_board()

    def input(self):
        GUI.POINT_QUEUE.queue.clear()
        x, y = GUI.POINT_QUEUE.get()
        return x, y

    def game_start(self, game_loop):
        loop_thread = threading.Thread(target=game_loop)
        loop_thread.setDaemon(True)
        loop_thread.start()
        self.tk.mainloop()


class TerminalUI(UI):
    """ 终端游戏 UI """
    name = 'TerminalUI'

    def __init__(self):
        super().__init__()
        self._init_board()

    def _init_board(self):
        for i in range(WIDTH):
            for j in range(HEIGHT):
                print('\t_', end='')
            print()

    def render(self, board, last_move):
        for i in range(WIDTH):
            for j in range(HEIGHT):
                print('\t{}'.format(
                    {BLACK: 'x', WHITE: 'o', EMPTY: '_'}[board[i, j]]
                ), end='')
            print()
        print()

    def message(self, message):
        print(message)

    def reset(self):
        pass

    def input(self):
        x, y = input('> ').split(',')
        x, y = int(x), int(y)
        return x, y


class HeadlessUI(UI):
    """ 无 UI """
    name = 'HeadlessUI'

    def __init__(self):
        super().__init__()

    def render(self, board, last_move):
        pass

    def message(self, message):
        pass

    def reset(self):
        pass

    def input(self):
        return -1, -1
