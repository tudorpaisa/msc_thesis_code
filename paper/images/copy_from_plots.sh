for i in ../../plots/baseline*; do
    cp $i ./baseline/
done

for i in ../../plots/performance_rnn*; do
    cp $i ./performance_rnn/
done

for i in ../../plots/c_rnn_gan*; do
    cp $i ./c_rnn_gan/
done
