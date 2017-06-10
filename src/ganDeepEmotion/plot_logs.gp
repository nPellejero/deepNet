# iter data_seen epoch c_val_loss c_val_acc c_test_loss c_test_acc
# iter data_seen epoch d_loss c_loss g_loss d_acc c_acc g_acc c_val_loss c_val_acc c_test_loss c_test_acc
reset
set title "Loss Function"
set xlabel "iter"
set ylabel "loss"
set style data lines
set logscale y
set yrange [*:*]
plot "errors.log" u 1:4 t "d_train",\
        "" u 1:5 t "g_train",\
        "" u 1:6 t "c_train"
