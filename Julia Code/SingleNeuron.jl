using DataFrames

cd("./data")

files = readdir()[2:4]

train = readtable(files[3], separator = ',')

val = readtable(files[2], separator = ',')

test = readtable(files[1], separator = ',')
