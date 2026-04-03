def read_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()  # Returns list of lines with \n
    return lines


def get_prompt(name):
    return read_file(f"prompts/{name}.txt")
