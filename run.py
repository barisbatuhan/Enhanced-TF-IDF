import argparse

from src.data.text_processor import TextProcessor, TextOps
from src.model.TF_IDF import TF_IDF

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
    TextOps.UNICODE, 
    TextOps.STOP_WORDS, 
    TextOps.LEMMATIZE, 
    TextOps.DIGITS
}

tr_corpus = args.tr_corpus
tr_data  = TextProcessor(tr_corpus, op_set, min_occur_cnt=10)

if args.val_corpus is not None:
    val_corpus = args.val_corpus
    val_data = TextProcessor(val_corpus, op_set, vocab=tr_data.vocab)

tf_idf  = TF_IDF(tr_data, ngram_range=(1, 1), max_df=0.95, min_df=0.05)
out  = tf_idf.fit_transform()

# val_out = tf_idf.fit_transform(val_data.get_docs())

print("--> Fit transform result:", out.shape)
print(out)

print("\n--> Feature Names:")
print(tf_idf.get_feature_names_out())