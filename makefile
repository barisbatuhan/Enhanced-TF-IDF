.SILENT: run test clean

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf **/__pycache__
	rm -rf */*/__pycache__
	rm -rf */*/*/__pycache__
	rm -rf .pytest_cache

run:
	python run.py -tc datasets/podcast_transcripts/processed/train.txt -vc datasets/podcast_transcripts/processed/valid.txt
	@make clean

test:
	pytest test.py
	@make clean