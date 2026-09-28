import re
import os

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        file_path = os.path.join(root, file)
        
        if all([item.isdecimal() for item in file.split(".")[0].split("_p")]):
            new_path = os.path.join(root, "Pixiv-" + file)
            try:
                os.rename(file_path, new_path)
            except:
                pass

        elif file.endswith(".jfif"):
            name = file.replace(".jfif", ".png")
            new_path = os.path.join(root, f"Twitter-{name}")
            try:
                os.rename(file_path, new_path)
            except:
                pass

        elif "Pixiv_" in file:
            new_file = file.replace("Pixiv_", "Pixiv-").replace("-p", "_p").replace("_Rotation-", "-Rotation_")
            new_path = os.path.join(root, new_file)
            try:
                os.rename(file_path, new_path)
            except:
                pass
            
        elif "Yande_" in file:
            new_file = file.replace("Yande_", "Yande-")
            new_path = os.path.join(root, new_file)
            try:
                os.rename(file_path, new_path)
            except:
                pass

        elif "Twitter_" in file:
            new_file = file.replace("Twitter_", "Twitter-")
            new_path = os.path.join(root, new_file)
            try:
                os.rename(file_path, new_path)
            except:
                pass

        elif "yande.re" in file:
            file_id = file.split(" ")[1]
            new_path = os.path.join(root, f"Yande-{file_id}." + file.split(".")[-1])
            try:
                os.rename(file_path, new_path)
            except:
                pass

_files = []

for _, _, files in  os.walk(r"E:\00.PROJEKT\hololive"):
    for file in files:
        _files.append(file)

for file in os.listdir(os.getcwd()):
    # print( os.getcwd())
    if file in _files:
        print(file)
        os.remove(file)