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
	python3 run.py -tc datasets/podcast_transcripts/processed/train.txt \
		-vc datasets/podcast_transcripts/processed/valid.txt \
		--lower --ascii --nodigit --nopunc --stop_words "#default" \
		--min_df 0.02 --max_df 0.98 --max_features 40 --visualize
	@make clean

test:
	pytest test.py
	@make clean