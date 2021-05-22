# multi-modal-qg
A Multi-Modal Question Generation Model.

## Steps to prepare data
First change pwd to dataset/
```
cd dataset
```

1. Assign question ids
	```
	python assign_question_id.py
	```
1. Get audio files from video files
	```
	python get_audio.py
	```
1. Samples salient audio clips
	```
	python get_salient_audioclips.py
	```
1. Samples salient frames
	```
	python get_salient_frames.py
	```
1. Samples salient text
	```
	python get_salient_text.py
	```
1. Now change back to project home
	```
	cd ..
	```
1. Get glove matrix
	```
	python get_glove_matrix.py
	```
1. Preprocess data
	```
	python preprocess_text.py
	```
1. Prepare splits and vocab
	```
	python prepare_data.py
	```

## Hyperparams
* Hyper parameters are read from config.py

## Train
```
python train.py
```



