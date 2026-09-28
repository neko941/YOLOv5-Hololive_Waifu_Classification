import os

def find_image_file(path_images, label):
  name = ".".join(label.split(".")[:-1])
  for image in os.listdir(path_images):
        if name == ".".join(image.split(".")[:-1]):
            return os.path.join(path_images, image)

def check_redundancy(train_img_root, train_label_root):
  for file in os.listdir(train_img_root):
    label = os.path.join(train_label_root, file.replace(file.split(".")[-1], "txt"))
    if not os.path.exists(label):
        print(f"Having: {file}")
        print(f"Not Exists: {label}", end="\n\n")

  for file in os.listdir(train_label_root):
    image = find_image_file(path_images=train_img_root, label=file)
    if image == None or not os.path.exists(image):
        print(f"Having: {file}")
        print(f"Not Exists: {image}", end="\n\n")

check_redundancy(
    train_img_root='images',
    train_label_root='labels')