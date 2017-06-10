#!/bin/bash
file=$1
tail -n +7 $file | awk '
BEGIN 	{ n=5;FS=" ";print "Iterations\tloss_lab\tloss_unl\ttrain_err\ttest_err" } 
		{
			NR>n;
			print $2 "\t" $8 "\t" $11 "\t" $15 "\t" $19
		}	
END   	{ print " - DONE -" } 
' 
