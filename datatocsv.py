source = open("german.data", "r")
target = open("german.csv", "a")

for line in source:
	for i in range(len(line)):
		if line[i] != " ":
			target.write(line[i])
		else:
			target.write(",")

source.close()
target.close()