import collections
import datetime
import os

""" Get classes from classes.txt """
classes = []
with open("classes.txt", "r") as f:
    for line in f:
        classes.append(line.strip())

""" Set Default values """
labels = collections.OrderedDict()
for i in range(len(classes)):
    labels[str(i)] = 0

""" Counting labels"""
for file in os.listdir("labels"):
    with open("labels/" + file, "r") as f:
        for line in f:
            labels[line.strip().split(" ")[0]] += 1

def padding(txt, indence=7*4):
    x = indence - len(txt)
    for i in range(x):
        txt = txt + " "    
    return txt        

def size_in_str(totalSize):
    totalSize = float(totalSize)
    if totalSize < 1024:
        return "{0:.0f} B".format(totalSize)
    totalSize = totalSize / 1024
    if totalSize < 1024:
        return "{0:.2f} KiB".format(totalSize)
    totalSize = totalSize / 1024
    if totalSize < 1024:
        return "{0:.2f} MiB".format(totalSize)
    totalSize = totalSize / 1024
    return "{0:.2f} GiB".format(totalSize)

num = len(os.listdir("images"))
total_size_images = [os.stat(f"images/{file}").st_size for file in os.listdir("images")]
now = datetime.datetime.now()

# with open("AFE-Info.txt", "wb") as f:
#     f.write(f"- QUANTITY: {num} images\n".encode())
#     f.write(f"- SIZE: {size_in_str(sum(total_size_images))}\n".encode())
#     f.write(f"- LAST UPDATE: {now.year}-{now.month}-{now.day}\n".encode())
#     f.write(f"- LABELS:\n".encode())
#     for i in range(len(classes)):
#         title = padding(f"{i}-{classes[i]}")
#         f.write(f"\t{title}:\t{labels[str(i)]}\n".encode())

print(f"- QUANTITY: {num} images")
print(f"- SIZE: {size_in_str(sum(total_size_images))}")
print(f"- LABELS:")
for i in range(len(classes)):
    title = padding(f"{i}-{classes[i]}")
    print(f"\t{title}:\t{labels[str(i)]}")