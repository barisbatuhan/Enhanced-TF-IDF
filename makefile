.SILENT: run test clean

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf **/__pycache__
	rm -rf */*/__pycache__
	rm -rf */*/*/__pycache__
	rm -rf .pytest_cache

run:
	python3 run.py
	@make clean

test:
	pytest test.py
	@make clean