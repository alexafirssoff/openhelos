from typing import Iterator, TypeVar, List

T = TypeVar('TFS')  # For generic types


def sliding_window_generator(data, window_size: int) -> Iterator[List[T]]:
    """
    Generates sliding windows (sub-lists) of a specified length from a list.

    Args:
      data: The source list.
      window_size: The length of each window.

    Yields:
      A sub-list (window) of length `window_size`.

    Raises:
      ValueError: If `window_size` is less than or equal to 0.
    """
    if window_size <= 0:
        raise ValueError("Размер окна (window_size) должен быть положительным.")
    
    # Determines up to which index iteration is needed for the last window to fit.
    # The last starting index = len(data) - window_size
    num_windows = len(data) - window_size + 1
    
    if num_windows <= 0:
        # If the window is larger than or equal to the list length, no windows can be formed,
        # or only one window can be formed if they are of equal length.
        if window_size == len(data) and len(data) > 0:
            yield data[:]  # Returns a copy of the entire list as a single window.
        return  # In other cases (window_size > len(data)), nothing is generated.
    
    for i in range(num_windows):
        # Returns a slice - the current window.
        yield data[i: i + window_size]


def sliding_window(sequence, window_size, step=1):
    """A sliding window generator that includes any remainder and uses a precise step."""
    seq_len = len(sequence)
    if seq_len == 0:
        return
    
    for i in range(0, seq_len, step):
        window = sequence[i:min(i + window_size, seq_len)]
        yield window


if __name__ == '__main__':
    # Example of use
    my_list = [1, 2, 3, 4, 5, 6, 7]
    window_len = 3
    
    print(f"Исходный список: {my_list}")
    print(f"Размер окна: {window_len}")
    
    print("Окна (генератор):")
    # Iterate over the generator
    for window in sliding_window_generator(my_list, window_len):
        print(window)
    # Output:
    # [1, 2, 3]
    # [2, 3, 4]
    # [3, 4, 5]
    # [4, 5, 6]
    # [5, 6, 7]
    
    # All windows can be collected into a list (if necessary and memory allows)
    all_windows = list(sliding_window_generator(my_list, window_len))
    # Output: Все окна списком: [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
    
    # Example: window larger than list
    for window in sliding_window_generator(my_list, 10):
        print(window)  # Nothing will be printed
    
    # Example: window equal to list
    for window in sliding_window_generator(my_list, 7):
        print(window)
    # Output: [1, 2, 3, 4, 5, 6, 7]
    
    # Example: empty list
    for window in sliding_window_generator([], 3):
        print(window)  # Nothing will be printed