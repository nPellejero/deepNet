# iter data_seen epoch c_val_loss c_val_acc c_test_loss c_test_acc
# iter data_seen epoch d_loss c_loss g_loss d_acc c_acc g_acc c_val_loss c_val_acc c_test_loss c_test_acc
reset
set title "Classification Error"
set xlabel "iter"
set ylabel "error rate"
set style data lines
set yrange [0.0:1.0]
plot "errors.log" u 1:7 t "c_val",\
        "" u 1:8 t "c_test",\
     "best.log" u 1:4 not with points,\
     "" u 1:5 not with points
