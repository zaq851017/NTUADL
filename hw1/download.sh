#!/usr/bin/env bash
glove=https://www.dropbox.com/s/fmopl98ccdngo8a/glove.6B.300d.txt?dl=1
embedding=https://www.dropbox.com/s/bzk0oz7klfdheq3/embedding.pkl?dl=1
model1=https://www.dropbox.com/s/fk6xsfrllljymii/model1.pkl?dl=1
model2_encoder=https://www.dropbox.com/s/0cbarsqo8y864dz/model2_encoder.pkl?dl=1
model2_decoder=https://www.dropbox.com/s/p5w9scqtkw1shhz/model2_decoder.pkl?dl=1
model3_encoder=https://www.dropbox.com/s/3qhnjo9dntrap3k/model3_encoder.pkl?dl=1
model3_decoder=https://www.dropbox.com/s/l79gjb69j0q6fbs/model3_decoder.pkl?dl=1

wget "${glove}" -O ./glove.6B.300d.txt
wget "${embedding}" -O ./embedding.pkl
wget "${model1}" -O ./model1.pkl
wget "${model2_encoder}" -O ./model2_encoder.pkl
wget "${model2_decoder}" -O ./model2_decoder.pkl
wget "${model3_encoder}" -O ./model3_encoder.pkl
wget "${model3_decoder}" -O ./model3_decoder.pkl