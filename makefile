.SILENT: run test clean clean-outputs

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf **/__pycache__
	rm -rf */*/__pycache__
	rm -rf */*/*/__pycache__
	rm -rf .pytest_cache

clean-outputs:
	rm -f *.csv *.png *.jpg

run:
	python run.py -tc datasets/podcast_transcripts/processed/train.txt \
		-vc datasets/podcast_transcripts/processed/valid.txt \
		--lower --ascii --nodigit --nopunc --stop_words "#default" \
		--min_df 0.05 --max_df 0.95 --max_features 100 --visualize
	@make clean

test:
	pytest test.py
	@make clean