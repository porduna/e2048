import sys
import time
import random
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

###########################################################################
#
#
#                          Core concepts
# 
# 


LINE_POSITIONS = range(16)

class Directions:
    left  = np.int8(0)
    right = np.int8(1)
    up    = np.int8(2)
    down  = np.int8(3)

    DIRECTIONS = left, right, up, down

    @staticmethod
    def name(direction):
        return {
            Directions.left  : 'left',
            Directions.right : 'right',
            Directions.up    : 'up',
            Directions.down  : 'down',
        }[direction]

def initial_value():
    """ Create an empty board """
    return np.int64(0)

def initial_array():
    return build_array(initial_value())

def build_array(value):
    """ Given a np.int64, create a 2x2 np.array """
    remainder = value
    cur_data = np.array(LINE_POSITIONS, np.int8)

    for current in LINE_POSITIONS:
        cur_data[current] = remainder & 0xf
        remainder >>= 4

    return cur_data.reshape(4,4)

_REAL_VALUES = {
    0     : 0,
    2     : 1,
    4     : 2,
    8     : 3,
    16    : 4,
    32    : 5,
    64    : 6,
    128   : 7,
    256   : 8,
    512   : 9,
    1024  : 10,
    2048  : 11,
    4096  : 12,
    8192  : 13,
    16384 : 14,
    32768 : 15,
}

def _build_from_2048(arr):
    """ Given an array expressed in 2048 (e.g., 2, 4, 8, 16, 32, 64 ...),
    create a real one (where values are 0-15)
    """
    new_list = []
    for value in arr.reshape(16):
        new_list.append(_REAL_VALUES[value])

    return np.array(new_list, np.int8).reshape(4,4)

def build_value(arr):
    """ Given a 2x2 np.array, provide a np.int64 value """
    remainder = np.int64(0)
    sorted_arr = arr.reshape(16)[::-1]
    for value in sorted_arr[:15]:
        remainder |= value
        remainder <<= 4
    remainder |= sorted_arr[-1]
    return remainder

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



###############################################################################################
# 
# 
#        Simple movement methods
# 
# 


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
                if row[counter] and row[counter] == row[counter + 1]:
                    row[counter] += 1
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
        rotated_moved_arr = _only_move_left(rotated_arr)
        moved_arr = np.rot90(rotated_moved_arr, k = 3)
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
def potential_directions(arr):
    """ List of available directions to move """
    directions = []
    for direction in Directions.DIRECTIONS:
        if can_move_to(arr, direction):
            directions.append(direction)
    return directions

@requires_array
def can_move(arr):
    """ Can move in any diretion? True / False """
    return ( can_move_to(arr, Directions.left) or 
             can_move_to(arr, Directions.right) or 
             can_move_to(arr, Directions.up) or 
             can_move_to(arr, Directions.down) )


##################################################################
# 
# 
#                 Gap filling methods
# 

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
def _potential_fills(arr):
    hole_positions = _find_holes(arr, linear = True)
    holes = {
        # position, new_value : weight
    }
    for hole_position in hole_positions:
        # 1 is 2, 2 is 4
        holes[hole_position, 1] = 1.0 / len(hole_positions) * 0.9
        holes[hole_position, 2] = 1.0 / len(hole_positions) * 0.1
    return holes

@requires_array
def potential_states_to(arr, direction):
    direction_moves = {
        # value1 : chances
    }

    moved_arr = _only_move(arr, direction)
    if (arr == moved_arr).all():
        # If the arrays are equal, skip
        return

    fills = _potential_fills(moved_arr)
    for (position, new_value), chances in fills.items():
        new_arr = np.copy(moved_arr)
        new_arr[position / 4, position % 4] = new_value
        cur_value = build_value(new_arr)
        direction_moves[cur_value] = chances

    return direction_moves


@requires_array
def potential_states(arr):
    all_potential_moves = {
        # Directions.left : {
        #      all the potential new states when left is selected:
        # 
        #      value1 : chances,
        #      value2 : chances,
        #      value3 : chances,
        # 
        #      e.g.,
        #      0x0010 : 0.75, 
        # }
    }

    for direction in Directions.DIRECTIONS:
        direction_moves = potential_states_to(arr, direction)
        if direction_moves:
            all_potential_moves[direction] = direction_moves

    return all_potential_moves

@requires_array
def _fill_random(arr):
    positions_fills = _potential_fills(arr)
    if not positions_fills:
        raise Exception("Couldn't move there")

    position = np.random.choice(range(len(positions_fills.keys())), p = positions_fills.values())

    new_position, new_value = positions_fills.keys()[position]

    arr[new_position / 4, new_position % 4] = new_value

@requires_array
def initialize_board_random(arr):
    _fill_random(arr)
    _fill_random(arr)

@requires_array
def move_random(arr, direction):
    moved_arr = _only_move(arr, direction)
    _fill_random(moved_arr)
    return moved_arr

###############################################################################
# 
# 
#                 Board object
# 
# 

class Board(object):

    """ Object oriented version of the functions above """

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

    def potential_directions(self):
        return potential_directions(self.arr)

    def move_random(self, direction):
        new_arr = move_random(self.arr, direction)
        return Board.fromarray(new_arr)

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


######################################################
# 
# 
#                  Simulator
# 

class GameSimulator(object):
    """ Subclass this class to have a simulator. 
    You can either subclass the next_direction() method
    or the nextBoard(board) method, depending on your constraints.
    By default it makes very few operations, but there are many
    available. """

    def __init__(self, stdout = None, measuring = False):
        self.stdout       = stdout
        self.current_arr  = initial_array()
        self.max_value    = 0
        self.measurements = []
        self.measuring    = measuring

    def start(self):
        initialize_board_random(self.current_arr)

    def print_current_state(self):
        if self.stdout:
            print >> self.stdout, self.current_board

    def run(self):
        self.max_value = self.current_arr.max()
        self.print_current_state()

        initial_time = time.time()
        movements = 0

        while can_move(self.current_arr):
            value = build_value(self.current_arr)
            if self.measuring:
                before = time.time()
            direction = self.next_direction()
            if self.measuring:
                after = time.time()
                self.measurements.append( (after - before) )

            if self.stdout:
                print >> self.stdout
                print >> self.stdout, "    Moving to %s" % Directions.name(direction)
                print >> self.stdout
            self.current_arr = move_random(self.current_arr, direction)
            self.max_value = self.current_arr.max()
            self.print_current_state()
            movements += 1

        final_time = time.time()

        if self.stdout:
            print >> self.stdout, "Max value: %s" % ( 2 ** self.max_value )
            print >> self.stdout, "Total time: %.3f seconds" % (final_time - initial_time)
            print >> self.stdout, "%s movements" % movements

    @property
    def current_value(self):
        return build_value(self.current_arr)

    @property
    def current_board(self):
        return Board.fromarray(self.current_arr)

    @property
    def directions(self):
        return potential_states(self.current_arr)

    def next_direction(self):
        """ Override me """
        pass

class RandomGameSimulator(GameSimulator):
    def next_direction(self):
        return random.choice(self.directions.keys())

class HumanGameSimulator(GameSimulator):
    KEYS = {
            'w' : Directions.up,
            's' : Directions.down,
            'a' : Directions.left,
            'd' : Directions.right,
        }

    def start(self):
        print "Play with 'w', 'a', 's', 'd'. Exit with 'e' or 'q'"
        return super(HumanGameSimulator, self).start()

    def next_direction(self):
        # pip install py-getch
        import getch
        while True:
            from getch.getch import getch
            
            if self.stdout != sys.stdout:
                print self.current_board

            where = getch()
            if where in HumanGameSimulator.KEYS:
                return HumanGameSimulator.KEYS[where]

            if where in ('e','q'):
                raise Exception("Finish requested by user")


def _main():
    simulator = RandomGameSimulator(open('simulator.txt', 'w'))
    simulator.start()
    simulator.run()

    # game = HumanGameSimulator(open('simulator-2.txt', 'w'))
    # game.start()
    # game.run()


if __name__ == '__main__':
    _main()
