reset
set terminal png
set output "grafico.png"
set style data lines
set key right

set title "Train loss labeled, Train loss unlab, train error, test error vs. training iter."
set xlabel "Training iterations"
set logscale y
#set yrange [0:5]
set ylabel "loss"
plot "salidaParsed.txt" using 1:2 title "loss_lab" , "salidaParsed.txt" using 1:3 title "loss_unlab", "salidaParsed.txt" using 1:4 title "train_error"  , "salidaParsed.txt" using 1:5 title "test_error"




