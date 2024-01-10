with open("f.txt") as f:
    for line in f:
        # print(line)
        print(f"{line.split()[0]},",end="")
print()
with open("f.txt") as f:
    for line in f:
        # print(line)
        print(f"{line.split()[1]},",end="")