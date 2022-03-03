import os


def listTranscripts(pathDB, recursionLevel, suffix):

    dirList = [pathDB]
    for recursion in range(recursionLevel):
        nextList = []
        for item in dirList:
            nextList += [
                os.path.join(item, f)
                for f in os.listdir(item)
                if os.path.isdir(os.path.join(item, f))
            ]
        dirList = nextList

    sizeSuffix = len(suffix)
    transcripts = []
    for dir in dirList:
        transcripts += [
            os.path.join(dir, x) for x in os.listdir(dir) if x[-sizeSuffix:] == suffix
        ]
    print(f"{pathDB} : {len(transcripts)} files found")

    return transcripts


def mergeTranscripts(path1, path2, dest):

    with open(path1, "r") as f1:
        data1 = f1.readlines()

    with open(path2, "r") as f2:
        data2 = f2.readlines()

    fullData = data1 + data2
    with open(dest, "w") as f3:
        for item in fullData:
            f3.write(item)


def mergeAll(toMerge, pathDB1, pathDB2, pathDBOut):

    for item in toMerge:
        path1 = os.path.join(pathDB1, item)
        path2 = os.path.join(pathDB2, item)
        pathOut = os.path.join(pathDBOut, item)

        mergeTranscripts(path1, path2, pathOut)


l1 = "/checkpoint/mriviere/LibriVox_tests/9h/"
l2 = "/checkpoint/mriviere/LibriVox_tests/1h/"
dest = "/checkpoint/mriviere/LibriVox_tests/10h/"
s1 = len(l1)

data_9h = listTranscripts(l1, 2, ".txt")
data_1h = listTranscripts(l2, 2, ".txt")


data_9h = [x[s1:] for x in data_9h]
data_1h = [x[s1:] for x in data_1h]

common_data = set(data_9h).intersection(set(data_1h))
print(common_data)

mergeAll(common_data, l1, l2, dest)
