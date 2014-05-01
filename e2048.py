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
    for value in arr.flat:
        new_list.append(_REAL_VALUES[value])

    return np.array(new_list, np.int8).reshape(4,4)

def build_value(arr):
    """ Given a 2x2 np.array, provide a np.int64 value """
    remainder = np.int64(0)
    sorted_arr = arr.flat[::-1]
    for value in sorted_arr[:15]:
        remainder |= value
        remainder <<= 4
    remainder |= sorted_arr[-1]
    return remainder

ACTIVATE_REQUIRES = False
if __debug__ and ACTIVATE_REQUIRES:
    def requires_value(func):
        """ Decorator: any function with this decorator, the
        first value will be converted to a np.int64 """

        @functools.wraps(func)
        def wrapper(value, *args, **kwargs):
            if not isinstance(value, np.int64):
                raise TypeError("%s should be an np.int64" % value)
            return func(value, *args, **kwargs)
        return wrapper

    def requires_array(func):
        """ Decorator: any function with this decorator, the
        first value will be converted to a 2x2 array """

        @functools.wraps(func)
        def wrapper(arr, *args, **kwargs):
            if not isinstance(arr, np.ndarray):
                raise TypeError("%s should be an np.int64" % arr)
            return func(arr, *args, **kwargs)
        return wrapper
else:
    def requires_value(func):
        return func

    def requires_array(func):
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


# 
# The following "can move" is long in code, but still efficient
# (compared to structures / indexing according to the profiler)
# 

@requires_array
def can_move_to(arr, direction):
    if direction == Directions.left:

        for row in arr:
            a, b, c, d = row
            # If they're all zero, forget about this row
            if 0 == a == b == c == d:
                continue

            # If there is a zero with something on the right
            if d:
                if d == c or a == 0 or b == 0 or c == 0:
                    return True
            elif c:
                if c == b or a == 0 or b == 0:
                    return True
            elif b:
                if b == a or a == 0:
                    return True

        return False 

    if direction == Directions.right:

        for row in arr:
            a, b, c, d = row
            # If they're all zero, forget about this row
            if 0 == a == b == c == d:
                continue

            # If there is a zero with something on the right
            if a:
                if a == b or b == 0 or c == 0 or d == 0:
                    return True
            elif b:
                if b == c or c == 0 or d == 0:
                    return True
            elif c:
                if c == d or d == 0:
                    return True

        return False 

    if direction == Directions.up:

        for i in 0,1,2,3:
            col = arr[:,i]
            a, b, c, d = col
            # If they're all zero, forget about this row
            if 0 == a == b == c == d:
                continue

            # If they're equal or there is a zero below
            if d:
                if d == c or a == 0 or b == 0 or c == 0:
                    return True
            elif c:
                if c == b or a == 0 or b == 0:
                    return True
            elif b:
                if b == a or a == 0:
                    return True

        return False 

    if direction == Directions.down:

        for i in 0,1,2,3:
            col = arr[:,i]
            a, b, c, d = col
            # If they're all zero, forget about this row
            if 0 == a == b == c == d:
                continue

            # If there is a zero with something on the right
            if a:
                if a == b or b == 0 or c == 0 or d == 0:
                    return True
            elif b:
                if b == c or c == 0 or d == 0:
                    return True
            elif c:
                if c == d or d == 0:
                    return True

        return False 

    raise Exception("Invalid direction")

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
    return zip(*np.where(arr == 0))

@requires_array
def _potential_fills(arr):
    hole_positions = zip(*np.where(arr == 0))
    holes = {
        # position, new_value : weight
    }
    for hole_position in hole_positions:
        hashable_position = tuple(hole_position)
        # 1 is 2, 2 is 4
        holes[hashable_position, 1] = 1.0 / len(hole_positions) * 0.9
        holes[hashable_position, 2] = 1.0 / len(hole_positions) * 0.1
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
        new_arr[position] = new_value
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

    arr[new_position] = new_value

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
    
    @property
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

class GameStrategy(object):
    """ Subclass this class to have a simulator. 
    You can either subclass the next_direction() method
    or the nextBoard(board) method, depending on your constraints.
    By default it makes very few operations, but there are many
    available. """

    def __init__(self, stdout = None, measuring = False, dont_start = False):
        self.stdout       = stdout
        self.current_arr  = initial_array()
        self.max_value    = 0
        self.measurements = []
        self.measuring    = measuring
        self.total_time   = 0
        self.movements    = 0

        if not dont_start:
            self.start()

    def start(self):
        initialize_board_random(self.current_arr)
        self.initialize()

    def initialize(self):
        pass

    def print_current_state(self):
        if self.stdout:
            print >> self.stdout, self.current_board

    def run(self):
        self.max_value = self.current_arr.max()
        self.print_current_state()

        initial_time = time.time()

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
            self.movements += 1

        final_time = time.time()
        self.total_time = final_time - initial_time

        if self.stdout:
            print >> self.stdout, "Max value: %s" % ( 2 ** self.max_value )
            print >> self.stdout, "Total time: %.3f seconds" % self.total_time
            print >> self.stdout, "%s movements" % self.movements

    @property
    def current_value(self):
        return build_value(self.current_arr)

    @property
    def current_board(self):
        return Board.fromarray(self.current_arr)

    @property
    def potential_directions(self):
        return potential_directions(self.current_arr)

    @property
    def potential_states(self):
        return potential_states(self.current_arr)

    def next_direction(self):
        """ Override me """
        pass


#
# 
# 
#           Sample dummy strategies
# 
# 


class RandomGameStrategy(GameStrategy):
    def next_direction(self):
        return random.choice(self.potential_directions)

class FirstDirectionGameStrategy(GameStrategy):
    def next_direction(self):
        return self.potential_directions[0]

class HumanGameStrategy(GameStrategy):
    KEYS = {
            'w' : Directions.up,
            's' : Directions.down,
            'a' : Directions.left,
            'd' : Directions.right,
        }

    def initialize(self):
        print "Play with 'w', 'a', 's', 'd'. Exit with 'e' or 'q'"

    def next_direction(self):
        # pip install py-getch
        import getch
        while True:
            from getch.getch import getch
            
            if self.stdout != sys.stdout:
                print self.current_board

            where = getch()
            if where in HumanGameStrategy.KEYS:
                return HumanGameStrategy.KEYS[where]

            if where in ('e','q'):
                raise Exception("Finish requested by user")


#################################################################
# 
# 
#     Strategy Runner
# 
# 
# 

class StrategyRunner(object):
    def __init__(self, SimulatorClass, iterations = 1000):
        self.klass = SimulatorClass
        self.iterations = iterations

    def run(self):
        movements  = []
        total_time = []
        max_values = []

        initial_time = time.time()
        for _ in xrange(self.iterations):
            simulator = self.klass()
            simulator.run()

            movements.append(simulator.movements)
            total_time.append(simulator.total_time)
            max_values.append(simulator.max_value)
        end_time = time.time()

        total_time_all_iterations = end_time - initial_time
        
        print
        print "Simulator %s executed %s times (%s movements) in %.2f seconds " % (self.klass.__name__, self.iterations, np.array(movements).sum(), total_time_all_iterations)
        print
        self._show_variable_summary("movements", movements)
        self._show_variable_summary("total_time", total_time)

        max_values = np.array(max_values)

        print " * Max values:"
        print "   - max: %s (%s)" % (max_values.max(), 2 ** max_values.max() )
        print "   - mean: %s" % max_values.mean()
        print "   - std: %s" % max_values.std()
        print "   - min: %s (%s)" % ( max_values.min(), 2 ** max_values.min() )
        print

    def _show_variable_summary(self, name, distribution):
        distribution = np.array(distribution)

        print " * %s:" % name.title()
        print "   - max: %s" % distribution.max()
        print "   - mean: %s" % distribution.mean()
        print "   - std: %s" % distribution.std()
        print "   - min: %s" % distribution.min()


class FutureCalculator(object):

    """ Given a board and a number of steps, calculate ahead all the 
    potential combinations. 
    
    This class has its own stored step, so you
    should keep it and call the next_step(direction) method so discards
    the other possibilities and calculates the next step, without 
    re-calculating all the intermediate possibilities.
    
    It also makes it possible to provide a function and evaluate it on
    the last leafs of the tree.
    """
    def __init__(self, board, steps, initial = 0, max_value = 16):
        # Basic variables
        self.board     = board
        self.size      = size
        self.max_value = max_value

        # Structures:
        #   
        #   Recursive structure that stores all the possibilities. Whenever a new step
        #   is done, the tree becomes one of its branches (e.g., self._tree = self._tree[dir]),
        #   and on top of the new tree, a new generation is calculated.
        # 
        self._tree = {
            # direction1 : {
            #    value1 : {
            #        'p' : 0.9, # or 0.1: local probability,
            #        'f' : False, # finished or not (e.g., when running "next", do not go through
            #                     # this branch)
            #        direction1 : {
            #        }
            #    },
            # } 
        }

        # Cache: 
        #   
        #   Many positions are repeated. For instance if you move left and then right
        #   or you move right and then left, you will get the same structure. So as 
        #   to avoid recalculating them, they are all stored here, indexed first by the
        #   summatory of all the values which are inside the board. This way, whenever
        #   all the leafs have passed a new value, all the previous values can be removed
        #   from this cache. If they're present in the tree they will obviously be 
        #   kept, but if they're not, that memory will automatically be restored.
        #   
        #   While the leafs will typically have a similar values, the random of being
        #   2 or 4 makes this impossible. However, the cache shouldn't be too big given
        #   that most of its structures are already in the tree.
        #   
        self._cache = {
            # summatory : {
            #     board_value1 : complete_subtree,
            #     board_value2 : complete_subtree,
            #     board_value3 : complete_subtree,
            # }
        }

    def next_step(self, direction):
        """ Moves the internal iterator one step forward. """
        pass

    def evaluate(self, func, summary = None):
        """ Walk through the leafs of the tree, running func(value). It returns a dictionary
        in the following form:

        {
            direction1 : {
                'finished' : {
                    value1: probabilities (e.g., 0.3),
                    value2: probabilities (e.g., 0.2),
                },
                'running' : {
                    # value1 or value2 are the values returned by func
                    value1 : probabilities (e.g., 0.3),
                    value2 : probabilities (e.g., 0.2),
                }
            },
            direction 2: {
                # Same
            },
        }

        If "summary" is a number, then this value is returned:

        {
            direction1 : {
                0: probabilities (e.g., 0.3),
                1: probabilities (e.g., 0.2),
            },
            direction 2: {
                # Same
            },
        }

        Where 0 means "sum of all the possibilities < value" and 1 means > value.

        Note: the constructor of FutureCalculator has a "max_value", which defaults to 16 which
        is impossible to achieve. If you set it to 11 (i.e., 2048), whenever a board achieves this
        point it becomes "finished". Otherwise, FutureCalculator will not consider one game finished
        until it has achieved the condition where it can't be moved, regardless value achieved.
        """
        evaluation = {}

        # Magic ;-)

        if summary:
            return self.summarize(evaluation, summary)
        else:
            return evaluation

    def summarize(self, evaluation, min_value):
        """ Given the result of evaluate(), sum all the information to obtain the following structure: 
{
            direction1 : {
                0: probabilities (e.g., 0.3),
                1: probabilities (e.g., 0.2),
            },
            direction 2: {
                # Same
            },
        }

        Where 0 means "sum of all the possibilities < value" and 1 means > value.
        """

def _main():
    pass

    # simulator = RandomGameStrategy(open('simulator.txt', 'w'))
    # simulator.run()

#    game = HumanGameStrategy(open('simulator-2.txt', 'w'))
#    game.run()

    tester = StrategyRunner(RandomGameStrategy)
    tester.run()

#    tester = StrategyRunner(FirstDirectionGameStrategy)
#    tester.run()

if __name__ == '__main__':
    _main()

