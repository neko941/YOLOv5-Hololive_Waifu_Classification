import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, default="0")
    parser.add_argument('-i', '--input', type=str, default="labels")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    root = opt.input

    for file in os.listdir(root):
        content = open(os.path.join(root, file), "r")
        lines = content.readlines()
        temp_lines = []

        for line in lines:
            components = line.split(" ")
            components[0] = opt.label
            temp_lines.append(" ".join(components))
            
            content = open(os.path.join(root, file), "w")
            for temp_line in temp_lines:
                content.write(temp_line)