usage:
	@echo eliza -- run Twitter bot: Eliza
	@echo train -- resume the training for the Language-Model
	@echo test -- test the Language-Model

eliza:
	python3 ./eliza.py

test:
	python3 ./gen_test.py eliza.model

train:
	python3 ./train.py -i data/all.txt -o eliza.model --iteration 10000000 --resume
