import sys
import functools
import numpy as np

"""
Efficient 2048 game for playing with machine learning.

One board in 2048 is represented by 16 squares of values 0-15 (in particular,
2^[0-15], representing 0 = empty). That's 4 bits per square, 16 x 4 = 64 bits
required. A np.int64 is a perfect structure to represent a board: it's simple,
immutable, hashable and can be used as key for dicts. For this reason, this game
only uses np.int64 structures when storing boards. This way it is possible to
store millions of boards even in memory.

The board can be interpreted with the Board class. It provides the basic
movements for simulating a game, or for extracting all the potential next steps
and the possibilities of each happening, as well as debugging tools (e.g.,
printing it). For efficiency purposes (not requiring to create an object), all
its methods are provided as functions.
"""

LINE_POSITIONS = range(16)

class Directions:
    left  = 'left'
    right = 'right'
    up    = 'up'
    down  = 'down'

    DIRECTIONS = left, right, up, down

def initial_value():
    """ Create an empty board """
    return np.int64(0)

def build_array(value):
    """ Given a np.int64, create a 2x2 np.array """
    remainder = value
    cur_data = np.array(LINE_POSITIONS, np.int8)

    for current in LINE_POSITIONS:
        cur_data[current] = remainder & 0xf
        remainder >>= 4

    return cur_data.reshape(4,4)

def build_value(arr):
    """ Given a 2x2 np.array, provide a np.int64 value """
    remainder = np.int64(0)
    sorted_arr = arr.reshape(16)[::-1]
    for value in sorted_arr[:15]:
        remainder |= value
        remainder <<= 4
    remainder |= sorted_arr[-1]
    return remainder

if __debug__:
    def requires_value(func):
        """ Decorator: any function with this decorator, the
        first value will be converted to a np.int64 """

        @functools.wraps(func)
        def wrapper(value_or_arr, *args, **kwargs):
            if isinstance(value_or_arr, np.int64):
                return func(value_or_arr, *args, **kwargs)
            else:
                return func(build_value(value_or_arr), *args, **kwargs)
        return wrapper

    def requires_array(func):
        """ Decorator: any function with this decorator, the
        first value will be converted to a 2x2 array """

        @functools.wraps(func)
        def wrapper(value_or_arr, *args, **kwargs):
            if isinstance(value_or_arr, np.int64):
                return func(build_array(value_or_arr), *args, **kwargs)
            else:
                return func(value_or_arr, *args, **kwargs)
        return wrapper
else:
    def requires_value(func):
        return func


@requires_value
def board_print(value, output = sys.stdout, before = '', numbers = False, blank = '_'):
    arr = build_array(value)

    output_value = ''
    for row in arr:
        output_value += before
        for y in row:
            if numbers:
                real_value = str(y)
            elif y == 0:
                real_value = blank
            else:
                real_value = str(2 ** y)

            # Always assume 2048
            real_value = (' ' * (4 - len(real_value))) + real_value 
            output_value += "%s " % real_value
        output_value += "\n"
    if output is not None:
        print >> output, output_value
    return output_value

@requires_array
def _find_holes(arr, linear = False):
    positions_lineal = np.where(arr.reshape(16) == 0)[0]
    if linear:
        return positions_lineal

    return _linear_to_array(positions_lineal)

def _linear_to_array(positions_lineal):
    positions_arr = []
    for position_lineal in positions_lineal:
        row = position_lineal / 4
        col = position_lineal % 4
        positions_arr.append((row, col))
    return positions_arr

@requires_array
def _move_left(arr):
    pass

@requires_array
def _can_move_left(arr, direction):
    pass

@requires_array
def can_move(arr, direction):
    # TODO
    if direction == Directions.left:
        target = arr
    elif direction == Directions.right:
        target = arr
    elif direction == Directions.up:
        target = arr
    else:
        target = arr

@requires_array
def can_move(arr):
    """ Can move in any diretion? True / False """
    return ( can_move(arr, Directions.left) or 
             can_move(arr, Directions.right) or 
             can_move(arr, Directions.up) or 
             can_move(arr, Directions.down) )

def only_move(value, direction):
    pass

def move_random(value, direction):
    pass

class Board(object):
    def __init__(self, value = 0):
        self._value = np.int64(value)
        self.arr   = build_array(value)

    def copy(self):
        return Board(self._value)

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        return isinstance(other, Board) and other.value == self.value

    def __hash__(self):
        return self._value
    
    def __repr__(self):
        return 'e2048.Board(%r)' % self.value

    def __str__(self):
        return board_print(self.value, output = None)


