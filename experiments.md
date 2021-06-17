
# Experiments log

### Exp-1 : 

### Exp-2 : 

### Exp-3 :

* Single layer lstm
* Greedy Decoding strategy
* Attention on context words

### Exp-4 :

* Single layer lstm
* Random Sampling Decode strategy
* Attention on context words

### Exp-5 :

* 2 layer lstm with dropout
* Random Sampling Decoder strategy
* Attention on context words
* Backprop to audio and video encoder

### Exp-6 :

* 3 layer lstm with dropout
* Attention on context words
* Backprop to audio and video encoder

### Exp-7 :

* Conv-LSTM based video encoder
* 3 layer lstm with dropout
* Attention on video frames
* Attention on context words

### exp-vqg_multi-1 :

* Prophetnet in eval mode
* 2 layer audio and video decoders
* Hidden sz - 512
* Generation head takes logits

### exp-vqg_multi-2 :

* Prophetnet in train mode
* 1 layer audio and video decoders
* Hidden sz - 256
* Generation head takes last hidden state
