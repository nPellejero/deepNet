# iter data_seen epoch c_val_loss c_val_acc c_test_loss c_test_acc
# iter data_seen epoch d_loss c_loss g_loss d_acc c_acc g_acc c_val_loss c_val_acc c_test_loss c_test_acc
reset
set style data lines
set xlabel "iter"

set term wxt 0
set title "Classification Accuracy"
set ylabel "Accuracy"
set yrange [*:*]
set grid y
plot "errors.log" u 1:(1-$7) t "c_val",\
        "" u 1:(1-$8) t "c_test",\
     "best.log" u 1:(1-$4) not with points,\
     "" u 1:(1-$5) not with points


set term wxt 1
set title "Loss Function"
set ylabel "loss"
set logscale y
set yrange [*:*]
plot "errors.log" u 1:4 t "d_train",\
        "" u 1:5 t "g_train",\
        "" u 1:6 t "c_train"
