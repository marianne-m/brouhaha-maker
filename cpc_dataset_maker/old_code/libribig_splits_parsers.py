import os
import sys
import json
import argparse
import progressbar
from phonem_parser import getPhoneTranscription
from common_voice_db_maker import (
    getPhoneConverter,
    applyConverter,
    savePhoneDict,
    replace_sil,
    strip_sil,
)
from pathlib import Path
from cpc.dataset import findAllSeqs


RAW_FOLD = {
    "ɚ": "ɚ",
    "ɪ": "ɪ",
    "ɛ": "ɛ",
    "iː": "i",
    "ᵻ": "ə",
    "ə": "ə",
    "ɐ": "ə",
    "ʊ": "ʊ",
    "ɔ": "ɔ",
    "ɔː": "ɔ",
    "oː": "o",
    "oːɹ": "o ɹ",
    "aɪ": "aɪ",
    "oʊ": "oʊ",
    "ɑː": "ɑː",
    "æ": "æ",
    "ɔːɹ": "ɔː ɹ",
    "ɑːɹ": "ɑ ɹ",
    "eɪ": "eɪ",
    "ɛɹ": "ɛ ɹ",
    "aɪɚ": "aɪ ɚ",
    "uː": "uː",
    "aʊ": "aʊ",
    "ɪɹ": "ɪ ɹ",
    "ɜː": "ə",
    "ʃ": "ʃ",
    "n": "n",
    "p": "p",
    "t": "t",
    "k": "k",
    "m": "m",
    "s": "s",
    "ʌ": "ʌ",
    "tʃ": "t ʃ",
    "d": "d",
    "əl": "l",
    "l": "l",
    "ɔɪ": "ɔɪ",
    "ɾ": "ɾ",
    "f": "f",
    "h": "h",
    "v": "v",
    "ŋ": "ŋ",
    "ð": "ð",
    "z": "z",
    "w": "w",
    "ɡ": "ɡ",
    "dʒ": "d ʒ",
    "ɹ": "ɹ",
    "b": "b",
    "i": "i",
    "j": "j",
    "θ": "θ",
    "ʒ": "ʒ",
    "iə": "iə",
    "ʊɹ": "ʊɹ",
    "ʔ": "",
    "n̩": "n",
    "aɪə": "aɪ ə",
    "r": "ɹ",
    "x": "h",
    "()": "",
    "ææ": "æ æ",
    "ç": "h",
    "ɡʲ": "ɡ",
    "|ɹ": "ɹ",
    "ɑ̃": "ɑ n",
    "ɬ": "l",
    "o-ɹ": "o ɹ",
    "ɔ̃": "ɔ",
    "d-ʒ": "d ʒ",
    "t-ʃ": "t ʃ",
    "ɑ-ɹ": "ɑ ɹ",
    "ɛ-ɹ": "ɛ-ɹ",
    "ɔː-ɹ": "ɔ ɹ",
    "aɪ-ə": "aɪ ə",
    "ɪ-ɹ": "ɪ ɹ",
    "aɪ-ɚ": "aɪ ɚ",
}

BAD_PHONES = ["|ɛ", "|t", "|θ", "|f", "|n", "|ɪ", "|s", "|eɪ"]
FOLD = {key: val.split() for key, val in RAW_FOLD.items()}
CHAR_SIL = "|"

BAD_WORDS = ["riposted"]


def applyFoldToTranscript(textList, fold):
    out = []
    for item in textList:
        words = item.split()
        out_words = []
        for word in words:
            phones = [x for x in word.split("-") if x != ""]
            w_out = []
            for phone in phones:
                if phone not in fold:
                    print(phone)
                    print(fold)
                w_out += fold[phone]

            out_words.append("-".join(w_out))
        out.append(" ".join(out_words))
    return out


def removeBadSeq(seqList, textList, bad_phones=None, good_phones=None):
    outSeqs = []
    outText = []
    nREmoved = 0
    assert (bad_phones is not None) or (good_phones is not None)

    if len(seqList) != len(textList):
        raise ValueError("invalid size")
    nItems = len(seqList)
    for index in range(nItems):
        phones = textList[index].replace(" ", "").split("-")
        take = True
        for phone in phones:
            if bad_phones is not None and phone in bad_phones:
                take = False
                nREmoved += 1
            if good_phones is not None and phone not in good_phones:
                take = False
                nREmoved += 1

        if take:
            outSeqs.append(seqList[index])
            outText.append(textList[index])

    print(nREmoved)
    return outSeqs, outText


def transcriptsToPhone(pathTranscription, fold, converter, stats, bad_phones):

    with open(pathTranscription, "r") as file:
        data = file.readlines()

    textList = []
    seqList = []
    for line in data:
        line = line.replace("\n", "")
        vals = line.split()
        seqID = vals[0]
        text = " ".join(vals[1:]).lower()
        textList.append(text)
        seqList.append(seqID)

    phoneTranscriptions = getPhoneTranscription(textList, "en-us", backend="espeak")
    assert len(phoneTranscriptions) == len(seqList)
    phoneTranscriptions = applyFoldToTranscript(phoneTranscriptions, fold)
    seqList, phoneTranscriptions = removeBadSeq(
        seqList, phoneTranscriptions, bad_phones
    )
    phoneTranscriptions = replace_sil(phoneTranscriptions, CHAR_SIL)
    converter, stats = getPhoneConverter(phoneTranscriptions, converter, stats)
    intPhones = applyConverter(phoneTranscriptions, converter)

    nItems = len(phoneTranscriptions)
    return {seqList[x]: intPhones[x] for x in range(nItems)}, converter, stats


def getLexiconWords(pathLexicon, bad_words=BAD_WORDS):

    with open(pathLexicon, "r") as file:
        lines = file.readlines()

    return [x.split()[0] for x in lines if x.split()[0] not in BAD_WORDS]


def savePhoneLexicon(path_out, words, prononciation):
    assert len(words) == len(prononciation)
    N = len(words)

    with open(path_out, "w") as file:
        for i_ in range(N):
            p_ = prononciation[i_].replace("-", " ")
            file.write(f"{words[i_]} {p_}\n")


def getPhoneList(prononciations):
    out = set()
    for p_ in prononciations:
        out = out.union(set(p_.replace(" ", "-").split("-")))
    return out


def getFullPhonesTranscription(
    pathDB, recursionLevel, converter, stats, suffix, fold, bad_phones
):

    transcripts = [
        os.path.join(pathDB, x[1]) for x in findAllSeqs(pathDB, extension=suffix)[0]
    ]
    fullPhoneTranscript = {}
    print(f"{len(transcripts)} files found")

    bar = progressbar.ProgressBar(maxval=len(transcripts))
    bar.start()
    for index, transcript in enumerate(transcripts):
        bar.update(index)
        intPhones, converter, stats = transcriptsToPhone(
            transcript, fold, converter, stats, bad_phones
        )
        for key, val in intPhones.items():
            if key in fullPhoneTranscript:
                raise ValueError(f"Sequence {key} already has a transcript")
            fullPhoneTranscript[key] = val.split()

    bar.finish()
    return fullPhoneTranscript, converter, stats


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Trainer")
    subparsers = parser.add_subparsers(dest="command")

    parser_db = subparsers.add_parser("from_db")
    parser_db.add_argument("pathDB", type=str)
    parser_db.add_argument(
        "pathOut", type=str, help="Path to the output phone transcript (.txt)"
    )
    parser_db.add_argument("--file_extension", type=str, default=".wav")
    parser_db.add_argument(
        "--dataset_levels",
        type=int,
        default=2,
        help="Levels of recursion in the dataset",
    )
    parser_db.add_argument(
        "--path_phone_converter",
        type=str,
        default=None,
        help="Path to a phone convertion dictionnary if the phonemizer has already been run on another dataset",
    )

    parser_lexicon = subparsers.add_parser("lexicon")
    parser_lexicon.add_argument("path_in", type=str)
    parser_lexicon.add_argument("path_out", type=str)
    parser_lexicon.add_argument(
        "--path_phone_converter",
        type=str,
        default=None,
        help="Path to a phone convertion dictionnary if the phonemizer has already been run on another dataset",
    )
    return parser.parse_args(argv)


def loadjson(pathJSON):
    with open(pathJSON, "rb") as file:
        data = json.load(file)
    return data


def savejson(data, pathjson):
    with open(pathjson, "w") as file:
        json.dump(data, file, indent=2)


def main(argv):

    args = parse_args(argv)

    if args.command == "from_db":
        if args.path_phone_converter is not None:
            converter = loadjson(args.path_phone_converter)
        else:
            converter = {}

        stats = {}
        fullPhoneTranscript, converter, stats = getFullPhonesTranscription(
            args.pathDB,
            args.dataset_levels,
            converter,
            stats,
            "trans.txt",
            FOLD,
            BAD_PHONES,
        )

        pathOutTranscript = f"{os.path.splitext(args.pathOut)[0]}.txt"
        pathOutConverter = f"{os.path.splitext(args.pathOut)[0]}_converter.json"

        savePhoneDict(fullPhoneTranscript, pathOutTranscript)
        savejson(converter, pathOutConverter)
    else:
        lexicon = getLexiconWords(args.path_in)
        n_words = len(lexicon)

        print(f"{n_words} words found")

        shift = 0
        out_words, out_trans = [], []
        size_slice = 20000
        n_slices = (n_words - shift) // size_slice + 1

        converter = None
        if args.path_phone_converter is not None:
            converter = loadjson(args.path_phone_converter)

        for s_ in range(n_slices):

            print(f"Slice {s_} out of {n_slices}")

            p_start = min(n_words, shift + s_ * size_slice)
            p_end = min(n_words, p_start + size_slice)

            loc_lex_ = lexicon[p_start:p_end]
            # print(loc_lex_)

            phoneTranscriptions = getPhoneTranscription(
                loc_lex_, "en-us", backend="espeak"
            )
            phoneTranscriptions = applyFoldToTranscript(phoneTranscriptions, FOLD)

            try:
                words, trans = removeBadSeq(loc_lex_, phoneTranscriptions, BAD_PHONES)
            except ValueError:
                print(p_start, p_end, s_)
                sys.exit()
            trans = strip_sil(trans, CHAR_SIL)

            # trans = [x.replace(' ', '-') for x in trans]
            out_words += words
            out_trans += trans

            if s_ % 10 == 0:
                savePhoneLexicon(args.path_out, out_words, out_trans)

        savePhoneLexicon(args.path_out, out_words, out_trans)

        path_out_trans = (
            Path(args.path_out).parent / f"{Path(args.path_out).stem}_phones.json"
        )
        all_phones = getPhoneList(out_trans)
        print(list(all_phones))
        with open(path_out_trans, "w") as file:
            json.dump(list(all_phones), file, indent=2)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
