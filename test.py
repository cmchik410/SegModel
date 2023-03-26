import glob


A = glob.glob("datasets/ADE20K/images/training/*.jpg", recursive = True)

print(A)