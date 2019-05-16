#!/bin/bash
unzip baseline.zip -d ../models/
unzip baseline_50.zip -d ../models/

cat performance_rnn_part_* > performance_rnn.zip
unzip performance_rnn.zip -d ../models/

cat c_rnn_gan_part_* > c_rnn_gan.zip
unzip c_rnn_gan.zip -d ../models/

rm performance_rnn.zip
rm c_rnn_gan.zip
