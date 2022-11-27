import argparse
import numpy as np

from src.models import TfIdfModel
from src.types import TextOps
from src.utils import IO, Visualizer


parser = argparse.ArgumentParser(
    prog = 'TF-IDF Implementation',
    description = 'Given a corpus, program processes the text and runs a TF-IDF model.')

parser.add_argument('-tc', '--tr_corpus', type=str, required=True, help='txt file path of the train corpus.')
parser.add_argument('-vc', '--val_corpus', type=str, default=None, help='txt file path of the val corpus.')

parser.add_argument('--lower', action='store_true', help='Lower the texts if present.')
parser.add_argument('--nodigit', action='store_true', help='Removes digits from the texts, if present.')
parser.add_argument('--nopunc', action='store_true', help='Removes punctuations from the texts, if present.')
parser.add_argument('--ascii', action='store_true', help='Strip the texts with ascii, if present.')
parser.add_argument('--unicode', action='store_true', help='Strip the texts with unicode, if present.')
parser.add_argument('--lemmatize', action='store_true', help='Lemmatize the texts, if present.')
parser.add_argument('--stem', action='store_true', help='Stem the texts, if present.')
parser.add_argument(
    '--stop_words', type=str, default=None, 
    help="List the stop words with the format \"w1,w2,w3...,wn\" if they are custom, \
          to use the default words, pass '#default', to use none, do not add .")

parser.add_argument('--min_df', default=0.02, type=float, help='Filter ratio of min. occurring items. In range: [0, 1)')
parser.add_argument('--max_df', default=0.98, type=float, help='Filter ratio of max. occurring items. In range: (0, 1]')
parser.add_argument('--max_features', default=None, type=int, help='Maximum number of feature items to select.')

parser.add_argument('--visualize', action='store_true', help='Visualize the heatmap bad closeness of validation data, if present.')

args = parser.parse_args()

# Create the operation set from arguments

op_set = set()
op_info_txt = "[INFO] Running the script with ops -->"

if args.lower:
    op_info_txt += " Lower"
    op_set.add(TextOps.LOWER)

if args.nodigit:
    op_info_txt += " No_Digit"
    op_set.add(TextOps.DIGITS)

if args.ascii:
    op_info_txt += " ASCII_Strip"
    op_set.add(TextOps.ASCII)
if args.unicode:
    op_info_txt += " Unicode_Strip"
    op_set.add(TextOps.UNICODE)

if args.stop_words:
    op_info_txt += " Stop_Words"
    sw = args.stop_words.split(",")
    op_set.add(TextOps.STOP_WORDS)
else:
    sw = None

if args.nopunc:
    op_info_txt += " No_Punctuation"
    op_set.add(TextOps.PUNCTUATIONS)

if args.lemmatize:
    op_info_txt += " Lemmatize"
    op_set.add(TextOps.LEMMATIZE)
if args.stem:
    op_info_txt += " Stem"
    op_set.add(TextOps.STEM)

print(op_info_txt)
    
# Read the corpus from the files passed in arguments

tr_data = IO.read_txt_corpus(args.tr_corpus)
val_data = IO.read_txt_corpus(args.val_corpus) if args.val_corpus is not None else None

# Load the TF-IDF model

tf_idf  = TfIdfModel(
    op_set,
    analyzer="word",
    stop_words=sw,
    max_df=args.max_df, 
    min_df=args.min_df,
    max_features=args.max_features,
)

# Fit the model with the train data

out = tf_idf.train(tr_data)
feature_words = tf_idf.get_feature_names()

print("\n--> Feature Names: Size of", len(feature_words))
print(feature_words)

print("\n--> Stop Words:")
print(tf_idf.get_stop_words())

print("\n--> Saving fit transform result:", out.shape)
IO.save_to_csv(out, "train_result.csv", colnames=feature_words)

# Evaluate the model with validateion data

if val_data is not None:
    val_out = tf_idf.infer(val_data)
    print("\n--> Val transform result:", val_out.shape)
    IO.save_to_csv(out, "val_result.csv", colnames=feature_words)
    if args.visualize:
        Visualizer.vis_heatmap(val_out, "val_data_heatmap.png")
        Visualizer.vis_closeness(val_out, "val_data_closeness.png", labels=feature_words)