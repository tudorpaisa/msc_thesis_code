for i in *.tex; do
    sed -i 's/ymax=-0.5] {/ymax=-0\.5] {images\/c_rnn_gan\//g' $i
done
