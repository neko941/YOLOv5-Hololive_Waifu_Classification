import os

temp = []

def find_image_file(path_images, label):
  name = ".".join(label.split(".")[:-1])
  for image in os.listdir(path_images):
        if name == ".".join(image.split(".")[:-1]):
            return os.path.join(path_images, image)

for file in os.listdir('labels'):
    content = open(os.path.join('labels', file), "r")
    lines = content.readlines()
    temp_lines = []

    for line in lines:
        components = line.split(" ")
        if int(components[0]) > 6:
            temp.append([os.path.join('labels', file), file])
            
            img = find_image_file('images', file)
            temp.append([img, os.path.basename(img)])

for item in temp:
    os.rename(item[0], item[1])
