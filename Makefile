usage:
	@echo eliza -- run Twitter bot: Eliza
	@echo train -- resume the training for the Language-Model
	@echo test -- test the Language-Model

eliza:
	python ./eliza.py

test:
	python ./gen_test.py eliza.model

train:
	python ./train.py -i data/all.txt -o eliza.model --iteration 10000000 --resume
