import os
import subprocess
from progressbar import ProgressBar


def getPhoneTranscription(textList, languageCode, backend="espeak", chunk_size=100):

    n_chunks = len(textList) // chunk_size
    if len(textList) % chunk_size != 0:
        n_chunks += 1
    start = 0
    out = []

    bar = ProgressBar(maxval=n_chunks)
    bar.start()
    for i_ in range(n_chunks):
        bar.update(i_)
        out_trans = transcribe_chunk(
            textList[start : start + chunk_size], languageCode, backend
        )
        if len(out_trans) != len(textList[start : start + chunk_size]):
            print(out_trans)
            print(textList[start : start + chunk_size])
            raise RuntimeError(f"Invalid transcription {start,start+chunk_size}")
        out += out_trans
        start += chunk_size
    bar.finish()
    return out


def transcribe_chunk(textList, languageCode, backend="espeak"):
    pathTmpIn = "tmp.txt"
    pathTmpOut = "tmp_out.txt"
    with open(pathTmpIn, "w") as file:
        for item in textList:
            file.write(item + "\n")
            # file.write('\n')

    proc = subprocess.Popen(
        [
            "phonemize",
            "-b",
            backend,
            "-l",
            languageCode,
            pathTmpIn,
            "-o",
            pathTmpOut,
            "-p",
            "-",
            "--strip",
            "-j",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = proc.communicate()
    with open(pathTmpOut, "r") as file:
        data = file.readlines()
        output = [item.replace("\n", "") for item in data if len(item) > 0]
    # cmd = f"rm {pathTmpIn}"
    # os.system(cmd)
    # cmd = f"rm {pathTmpOut}"
    # os.system(cmd)
    return output
