import argparse
import numpy as np

from src.models import TfIdfModel
from src.utils import TextOps, IO, Visualizer

# python run.py -tc datasets/podcast_transcripts/processed/train.txt -vc datasets/podcast_transcripts/processed/valid.txt
# python run.py -tc datasets/amazon_clothing/processed/train.txt

parser = argparse.ArgumentParser(
    prog = 'TF-IDF Implementation',
    description = 'Given a corpus, program processes the text and runs a TF-IDF model.')

parser.add_argument('-tc', '--tr_corpus', type=str, required=True, help='txt file path of the train corpus.')
parser.add_argument('-vc', '--val_corpus', default=None, type=str, help='txt file path of the val corpus.')

args = parser.parse_args()

op_set = {
    TextOps.LOWER, 
    TextOps.ASCII, 
    TextOps.STOP_WORDS, 
    TextOps.LEMMATIZE, 
    TextOps.DIGITS,
    TextOps.PUNCTUATIONS
}

tr_data = IO.read_txt_corpus(args.tr_corpus)
val_data = IO.read_txt_corpus(args.val_corpus) if args.val_corpus is not None else None

tf_idf  = TfIdfModel(
    op_set,
    stop_words="default",
    max_df=0.98, 
    min_df=0.02,
    max_features=1000,
)
out = tf_idf.train(tr_data)
feature_words = tf_idf.get_feature_names()

print("--> Saving fit transform result:", out.shape)
IO.save_to_csv(out, "train_result.csv", colnames=feature_words)

if val_data is not None:
    val_out = tf_idf.infer(val_data)
    print("\n--> Val transform result:", val_out.shape)
    IO.save_to_csv(out, "val_result.csv", colnames=feature_words)
    Visualizer.vis_heatmap(val_out, "val_data_heatmap.png")
    Visualizer.vis_closeness(val_out, "val_data_closeness.png")

print("\n--> Feature Names: Size of", len(feature_words))
print(feature_words)

print("\n--> Stop Words:")
print(tf_idf.get_stop_words())