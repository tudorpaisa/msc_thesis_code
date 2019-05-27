mkdir archived

zip baseline.zip baseline
mv baseline.zip ./archived/

zip 2048_baseline_50.zip 2048_baseline_50
mv 2048_baseline_50.zip ./archived/

zip 2048_baseline.zip 2048_baseline
mv 2048_baseline.zip ./archived/

zip performance_rnn.zip performance_rnn_1_5
mv performance_rnn.zip ./archived/

zip c_rnn_gan.zip c_rnn_gan_1_5
mv c_rnn_gan.zip ./archived/

cd archived/
split -b 10M performance_rnn.zip "performance_rnn_part_"
split -b 10M c_rnn_gan.zip "c_rnn_gan_part_"
rm performance_rnn.zip
rm c_rnn_gan.zip
cd ../

mv archived/* ../saved_models/
rm -r archived
