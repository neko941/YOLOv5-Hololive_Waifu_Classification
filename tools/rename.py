import re
import os

folders = ["images", "labels"]

path_images = "images"
path_labels = "labels"

delete = []

for file in os.listdir(path_images):
    label = file.replace(file.split(".")[-1], "txt")
    if all([item.isdecimal() for item in file.split(".")[0].split("_p")]):
        if not os.path.exists(os.path.join(path_images, "Pixiv-" + file)):
            os.rename(os.path.join(path_images, file), os.path.join(path_images, "Pixiv-" + file))
            if os.path.exists(os.path.join(path_labels, label)):
                os.rename(os.path.join(path_labels, label), os.path.join(path_labels, "Pixiv-" + label))
        else:
            delete.append(os.path.join(path_images, file))
            if os.path.exists(os.path.join(path_labels, label)):
                delete.append(os.path.join(path_labels, label))

    elif "Pixiv_" in file:
        new_file = os.path.join(path_images, file.replace("Pixiv_", "Pixiv-").replace("-p", "_p").replace("_Rotation-", "-Rotation_"))
        if not os.path.exists(new_file):
            os.rename(os.path.join(path_images, file), new_file)
            if os.path.exists(os.path.join(path_labels, label)):
                new_label = os.path.join(path_labels, label.replace("Pixiv_", "Pixiv-").replace("-p", "_p").replace("_Rotation-", "-Rotation_"))
                os.rename(os.path.join(path_labels, label), new_label)
        else:
            delete.append(os.path.join(path_images, file))
            if os.path.exists(os.path.join(path_labels, label)):
                new_label = os.path.join(path_labels, label.replace("Pixiv_", "Pixiv-").replace("-p", "_p").replace("_Rotation-", "-Rotation_"))
                delete.append(new_label)
        
    elif "Yande_" in file:
        new_file = os.path.join(path_images, file.replace("Yande_", "Yande-"))
        os.rename(os.path.join(path_images, file), new_file)
        if os.path.exists(os.path.join(path_labels, label)):
            new_label = os.path.join(path_labels, label.replace("Yande_", "Yande-"))
            os.rename(os.path.join(path_labels, label), new_label)

    elif "Twitter_" in file:
        new_file = os.path.join(path_images, file.replace("Twitter_", "Twitter-"))
        os.rename(os.path.join(path_images, file), new_file)
        if os.path.exists(os.path.join(path_labels, label)):
            new_label = os.path.join(path_labels, label.replace("Twitter_", "Twitter-"))
            os.rename(os.path.join(path_labels, label), new_label)

    elif "yande.re" in file:
        file_id = file.split(" ")[1]
        os.rename(os.path.join(path_images, file), os.path.join(path_images, f"Yande_{file_id}." + file.split(".")[-1]))
        if os.path.exists(os.path.join(path_labels, label)):
            os.rename(os.path.join(path_labels, label), os.path.join(path_labels, f"Yande_{file_id}." + label.split(".")[-1]))

for file in delete:
    os.remove(file)