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


