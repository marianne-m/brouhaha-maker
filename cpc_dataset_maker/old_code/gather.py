import argparse
import sys
from pathlib import Path


def gather_files(file_list, path_out, ignore_header):
    out = []
    header = None if not ignore_header else ""
    for file_name in file_list:
        with open(file_name, "r") as file:
            lines = file.readlines()
            if header is None:
                header = lines[0]
            out += lines[1:]

    with open(path_out, "w") as file:
        if not ignore_header:
            file.write(header)
        for line in out:
            file.write(line)


def gather_results(path_out, world_size, ignore_header=False):

    base_name = Path(path_out).stem
    canditates = [x for x in Path(path_out).parent.glob(f"{base_name}*.txt")]
    shift = len(base_name)

    # Filter candidates
    out = []
    index = set()
    for x in canditates:
        data = x.stem[shift:].split("_")
        if len(data) == 3 and int(data[-1]) == world_size:
            out.append(x)
            index.add(int(data[-2]))

    # Check the number of candidates
    expected = set(range(world_size))
    missing = expected - index
    unexpected = index - expected

    if len(missing) == 0 and len(unexpected) == 0:
        gather_files(out, path_out, ignore_header)
        return True

    if len(missing) > 0:
        print(f"{len(missing)} elements missing : {missing}")

    if len(unexpected) > 0:
        print(f"{len(unexpected)} unexpected elements : {unexpected}")

    return False


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Linear separability trainer"
        " (default test in speaker separability)"
    )
    parser.add_argument("path_out", type=str, help="Path to the output file")
    parser.add_argument("world_size", type=int, help="Number of files to gather")
    parser.add_argument("--ignore_header", action="store_true")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    gather_results(args.path_out, args.world_size, args.ignore_header)
