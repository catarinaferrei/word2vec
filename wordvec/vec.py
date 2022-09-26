import fasttext
from logging import root
import os
import argparse
import subprocess
import sys
from shlex import quote
from fasttext import load_model



def convert_bin_to_vec(args):
    f = load_model(args.save)
    lines = []
    # get all words from model
    words = f.get_words()
    with open(args.vec, 'w') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass


def model(args):
    model = fasttext.train_unsupervised(input=args.src, model='cbow', lr=args.lr, dim=args.size, ws=args.window, epoch=args.iter)
    model.save_model(args.save)


def main ():
    parser = argparse.ArgumentParser(description="Create wordvec")
    word2vec_group = parser.add_argument_group(description="Create word2vec")
    word2vec_group.add_argument('--src', metavar='STR',help='Input file')
    word2vec_group.add_argument('--size', metavar='N', type=int, default=300, help='Dimensionality of the phrase embeddings (defaults to 300)')
    word2vec_group.add_argument('--window', metavar='N', type=int, default=5, help='Max skip length between words (defauls to 5)')
    word2vec_group.add_argument('--lr', metavar='N', type=float, help='learning rate')
    word2vec_group.add_argument('--iter', metavar='N', type=int, default=5, help='Number of training epochs (defaults to 5)')
    word2vec_group.add_argument('--save', help='output fild')
    word2vec_group.add_argument('--vec')

    args = parser.parse_args()
    model(args)
    convert_bin_to_vec(args)

if __name__ == '__main__':
    main()

