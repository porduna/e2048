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
def _only_move_left(arr):
    new_arr = np.copy(arr)
    for row in new_arr:
        if row.any():
            counter = 0
            while counter < 3 and row[counter:].any():
                while row[counter] == 0:
                    row[counter:3] = row[counter + 1:4]
                    row[3] = 0
                counter += 1
            
            counter = 0

            while counter < 3:
                if row[counter] == row[counter + 1]:
                    row[counter] *= 2
                    row[counter + 1: 3] = row[counter + 2:]
                    row[3] = 0
                counter += 1

    return new_arr

@requires_array
def _only_move(arr, direction):

    # Select movement operation before and after moving left
    if direction == Directions.left:
        return _only_move_left( arr )

    elif direction == Directions.right:
        rotated_arr = np.rot90(arr, k = 2)
        moved_arr = _only_move_left(rotated_arr)
        return np.rot90(moved_arr, k = 2)

    elif direction == Directions.up:
        rotated_arr = np.rot90(arr, k = 1)
        print "Original"
        print arr
        rotated_moved_arr = _only_move_left(rotated_arr)
        moved_arr = np.rot90(rotated_moved_arr, k = 3)
        print moved_arr
        return moved_arr
    
    elif direction == Directions.down:
        rotated_arr = np.rot90(arr, k = 3)
        moved_arr = _only_move_left(rotated_arr)
        return np.rot90(moved_arr, k = 1)
    else:
        raise Exception("Direction expected. Got: %s" % direction)

@requires_array
def can_move_to(arr, direction):
    new_arr = _only_move(arr, direction)
    original_value = build_value(arr)
    new_value = build_value(new_arr)
    return new_value != original_value

@requires_array
def can_move(arr):
    """ Can move in any diretion? True / False """
    return ( can_move_to(arr, Directions.left) or 
             can_move_to(arr, Directions.right) or 
             can_move_to(arr, Directions.up) or 
             can_move_to(arr, Directions.down) )

def only_move(value, direction):
    pass

def move_random(value, direction):
    pass

class Board(object):
    def __init__(self, value = 0):
        self._value = np.int64(value)
        self.arr   = build_array(value)

    def _only_move(self, direction):
        new_arr = _only_move(self.arr, direction)
        return Board.fromarray(new_arr)

    def can_move(self, direction = None):
        if direction is None:
            return can_move(self.arr)
        else:
            return can_move_to(self.arr, direction)

    def copy(self):
        return Board(self._value)

    @property
    def value(self):
        return self._value

    @staticmethod
    def fromarray(arr):
        return Board(build_value(arr))

    def __eq__(self, other):
        return isinstance(other, Board) and other.value == self.value

    def __hash__(self):
        return self._value
    
    def __repr__(self):
        return 'e2048.Board(%r)' % self.value

    def __str__(self):
        return board_print(self.value, output = None)

    def pretty_print(self, *args, **kwargs):
        return board_print(self.value, output = None, *args, **kwargs)

