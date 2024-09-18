from typing import Any, List


def write_to_file(arr: List[List[Any]], file_name: str) -> None:
    """Writes a 2D array to file_name with columns separated with tabs and rows with newlines"""

    with open(file_name, "w") as f:
        for row in arr:
            f.write("\t".join(map(str, row)) + "\n")


def read_to_array(file_name: str) -> List[List[Any]]:
    """Reads a file and returns a 2D array"""

    with open(file_name, "r") as f:
        return [line.strip().split("\t") for line in f]
