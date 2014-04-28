import numpy as np
import e2048

def check_building_arrays_equals(value):
    arr_built     = e2048.build_array(value)
    value_built = e2048.build_value(arr_built)
    print hex(value_built), hex(value)
    assert value_built == value

def test_build():
    initial_value   = e2048.initial_value()
    yield check_building_arrays_equals, initial_value

    for potential_value in 0x5, 0x3, 0xf0, 0x010101:
        yield check_building_arrays_equals, np.int64(potential_value)

    longest = np.int64(0xdeadbeef) << 32 | np.int64(0xdeadbeef)
    yield check_building_arrays_equals, longest

def test_print():
    value = np.int64(0x03)
    result = e2048.board_print(value)
    assert result == ( "   8    _    _    _ \n" 
                       "   _    _    _    _ \n"
                       "   _    _    _    _ \n"
                       "   _    _    _    _ \n")

    value = np.int64(0x03)
    result = e2048.board_print(e2048.build_array(value))
    assert result == ( "   8    _    _    _ \n" 
                       "   _    _    _    _ \n"
                       "   _    _    _    _ \n"
                       "   _    _    _    _ \n")

    result = e2048.board_print(value, before = '>>> ')
    assert result == ( ">>>    8    _    _    _ \n" 
                       ">>>    _    _    _    _ \n"
                       ">>>    _    _    _    _ \n"
                       ">>>    _    _    _    _ \n")


    value = np.int64(0x12345678)
    result = e2048.board_print(value)
    assert result == ( " 256  128   64   32 \n" 
                       "  16    8    4    2 \n"
                       "   _    _    _    _ \n"
                       "   _    _    _    _ \n")

def test_find_holes():
    value = np.int64(0x110101000101)
    # Positions not empty: 0, 2, 6, 8, 10, 11
    arr = e2048.build_array(value)
    holes = e2048._find_holes(arr)
    expected_positions = [
                    (0, 1),         (0, 3), 
            (1, 0), (1, 1),         (1, 3), 
                    (2, 1), 
            (3, 0), (3, 1), (3, 2), (3, 3)]
    assert expected_positions == list(holes)

    for hole in holes:
        assert arr[hole] == 0

    holes = e2048._find_holes(arr, linear = True)
    expected_positions = [
                    1,         3, 
            4,      5,         7, 
                    9, 
           12,     13,   14,  15]
    assert expected_positions == list(holes)

def test_only_move():

    initial_data = [
        [ 0, 0, 0, 0 ],
        [ 2, 2, 4, 8 ],
        [ 0, 0, 2, 2 ],
        [ 2, 2, 2, 2 ]
    ]

    expected_down = np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ],
        [ 0, 0, 4, 8 ],
        [ 4, 4, 4, 4 ]
    ])

    yield check_only_move, initial_data, expected_down, e2048.Directions.down

    expected_up = np.array([
        [ 4, 4, 4, 8 ],
        [ 0, 0, 4, 4 ],
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ]
    ])

    yield check_only_move, initial_data, expected_up, e2048.Directions.up

    expected_right = np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 4, 4, 8 ],
        [ 0, 0, 0, 4 ],
        [ 0, 0, 4, 4 ]
    ])

    yield check_only_move, initial_data, expected_right, e2048.Directions.right

    expected_left = np.array([
        [ 0, 0, 0, 0 ],
        [ 4, 4, 8, 0 ],
        [ 4, 0, 0, 0 ],
        [ 4, 4, 0, 0 ]
    ])

    yield check_only_move, initial_data, expected_left, e2048.Directions.left

def check_only_move(initial_data, expected, direction):

    arr = np.array(initial_data)
    board1 = e2048.Board.fromarray(arr)
    board2 = e2048.Board.fromarray(np.array(initial_data))

    moved = e2048._only_move(arr, direction)

    assert board1 == board2, "A new array is built"

    expected_board = e2048.Board.fromarray(expected)
    resulting_board = e2048.Board.fromarray(moved)

    assert expected_board == resulting_board

def test_can_move():

    # Test individuals
    no = [
        [ 2, 2, 2, 2 ],
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ]
    ]

    yield check_can_move, no, e2048.Directions.up

    no = [
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ],
        [ 0, 0, 0, 0 ],
        [ 2, 2, 2, 2 ]
    ]

    yield check_can_move, no, e2048.Directions.down

    no = [
        [ 2, 0, 0, 0 ],
        [ 2, 0, 0, 0 ],
        [ 2, 0, 0, 0 ],
        [ 2, 0, 0, 0 ]
    ]

    yield check_can_move, no, e2048.Directions.left

    no = [
        [ 0, 0, 0, 2 ],
        [ 0, 0, 0, 2 ],
        [ 0, 0, 0, 2 ],
        [ 0, 0, 0, 2 ]
    ]

    yield check_can_move, no, e2048.Directions.right



    # Test general
    no = [
        [ 2, 4, 6, 8 ],
        [ 8, 6, 4, 2 ],
        [ 2, 4, 6, 8 ],
        [ 8, 6, 4, 2 ]
    ]

    assert not e2048.can_move(np.array(no))


def check_can_move(no, direction):
    yes = np.array([
        [ 0, 0, 0, 0 ],
        [ 2, 2, 4, 8 ],
        [ 0, 0, 2, 2 ],
        [ 2, 2, 2, 2 ]
    ])
    
    assert e2048.can_move_to(yes, direction)

    assert not e2048.can_move_to(np.array(no), direction)

def test_basic_class():
    # Constructor
    board = e2048.Board(64)
    # Repr and == 
    assert board == eval(repr(board))
    
    # Setter
    # Str and setter working
    assert str(board) == str(e2048.Board(64))
    # getter
    assert board.value == 64
    
    # hash
    a = {}
    a[e2048.Board(3)] = 3
    a[e2048.Board(4)] = 5
    a[e2048.Board(4)] = 4 # Replace the previous one
    assert len(a) == 2 # Replaced correctly
    assert a[e2048.Board(4)] == 4
    assert a[e2048.Board(3)] == 3


