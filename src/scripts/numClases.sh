#!/bin/bash
#! para ejecutar en el directorio AULabels

SmileT=0
AU04T=0
AU02T=0
AU15T=0
TrackerfailT=0
AU18T=0
AU09T=0
negAU12T=0
AU10T=0
ExpressiveT=0
Unilateral_LAU12T=0
Unilateral_RAU12T=0
AU14T=0
Unilateral_LAU14T=0
Unilateral_RAU14T=0
AU05T=0
AU17T=0
AU26T=0
ForwardT=0
BackwardT=0
TotalLabsT=0
TotalDirsT=0
for label in $(ls 2new*)
do

Smile=0
AU04=0
AU02=0
AU15=0
Trackerfail=0
AU18=0
AU09=0
negAU12=0
AU10=0
Expressive=0
Unilateral_LAU12=0
Unilateral_RAU12=0
AU14=0
Unilateral_LAU14=0
Unilateral_RAU14=0
AU05=0
AU17=0
AU26=0
Forward=0
Backward=0
TotalLabs=0
IFS=' ' 
i=1
while read -r SmileF AU04F AU02F AU15F AU18F AU09F negAU12F AU10F ExpressiveF Unilateral_LAU12F Unilateral_RAU12F AU14F Unilateral_LAU14F Unilateral_RAU14F AU05F AU17F AU26F ForwardF BackwardF
do
test $i -eq 1 && ((i=i+1)) && continue
#	echo -e "Smile: $SmileF\n 04:$AU04F\n 02:$AU02F\n 15:$AU15F\n track:$TrackerfailF\n 18:$AU18F\n 09:$AU09F\n neg12:$negAU12F\n 10:$AU10F\n expressive:$ExpressiveF\n uniL12:$Unilateral_LAU12F\n uniR12:$Unilateral_RAU12F\n 14:$AU14F\n UniL14:$Unilateral_LAU14F\n UniR14:$Unilateral_RAU14F\n 05:$AU05F\n 17:$AU17F\n 26:$AU26F\n forward:$ForwardF\n back:$BackwardF\n"
		TotalLabs=$((TotalLabs + 1));
	 if [ "$SmileF" != "0.0" ] && [ "$SmileF" != "0" ]; then Smile=$((Smile + 1)); fi  
	 if [ "$AU04F" != "0.0" ] && [ "$AU04F" != "0" ]; then  AU04=$((AU04+1)); fi  
	 if [ "$AU02F" != "0.0" ] && [ "$AU02F" != "0" ]; then AU02=$((AU02+1)); fi  
	 if [ "$AU15F" != "0.0" ] && [ "$AU15F" != "0" ]; then AU15=$((AU15+1)); fi  
	 if [ "$AU18F" != "0.0" ] && [ "$AU18F" != "0" ]; then AU18=$((AU18+1)); fi  
	 if [ "$AU09F" != "0.0" ] && [ "$AU09F" != "0" ]; then AU09=$((AU09+1)); fi  
	 if [ "$negAU12F" != "0.0" ] && [ "$negAU12F" != "0" ]; then negAU12=$((negAU12+1)); fi  
	 if [ "$AU10F" != "0.0" ] && [ "$AU10F" != "0" ]; then AU10=$((AU10+1)); fi  
	 if [ "$ExpressiveF" != "0.0" ] && [ "$ExpressiveF" != "0" ]; then Expressive=$((Expressive+1)); fi  
	 if [ "$Unilateral_LAU12F" != "0.0" ] && [ "$Unilateral_LAU12F" != "0" ]; then Unilateral_LAU12=$((Unilateral_LAU12+1)); fi  
	 if [ "$Unilateral_RAU12F" != "0.0" ] && [ "$Unilateral_RAU12F" != "0" ]; then Unilateral_RAU12=$((Unilateral_RAU12+1)); fi  
	 if [ "$Unilateral_LAU14F" != "0.0" ] && [ "$Unilateral_LAU14F" != "0" ]; then Unilateral_LAU14=$((Unilateral_LAU14+1)); fi  
	 if [ "$Unilateral_RAU14F" != "0.0" ] && [ "$Unilateral_RAU14F" != "0" ]; then Unilateral_RAU14=$((Unilateral_RAU14+1)); fi  
	 if [ "$AU14F" != "0.0" ] && [ "$AU14F" != "0" ]; then AU14=$((AU14+1)); fi  
	 if [ "$AU05F" != "0.0" ] && [ "$AU05F" != "0" ]; then AU05=$((AU05+1)); fi  
	 if [ "$AU17F" != "0.0" ] && [ "$AU17F" != "0" ]; then AU17=$((AU17+1)); fi  
	 if [ "$AU26F" != "0.0" ] && [ "$AU26F" != "0" ]; then AU26=$((AU26+1)); fi  
	 if [ "$ForwardF" != "0.0" ] && [ "$ForwardF" != "0" ]; then Forward=$((Forward+1)); fi 
	 BackwardF=${BackwardF%?}  
	 if [ "$BackwardF" != "0.0" ] && [ "$BackwardF" != "0" ]; then  Backward=$((Backward+1)); fi  
done < $label
	SmileT=$((SmileT + Smile))
	AU04T=$((AU04T + AU04))
	AU02T=$((AU02T + AU02))
	AU15T=$((AU15T+AU15))
	AU18T=$((AU18T+AU18))
	AU09T=$((AU09T+AU09))
	negAU12T=$((negAU12T+negAU12))
	AU10T=$((AU10T+AU10))
	ExpressiveT=$((ExpressiveT+Expressive))
	Unilateral_LAU12T=$((Unilateral_LAU12T+Unilateral_LAU12))  
	Unilateral_RAU12T=$((Unilateral_RAU12T+Unilateral_RAU12))  
	Unilateral_LAU14T=$((Unilateral_LAU14T+Unilateral_LAU14))  
	Unilateral_RAU14T=$((Unilateral_RAU14T+Unilateral_RAU14)) 
	AU14T=$((AU14T+AU14))
	AU05T=$((AU05T+AU05))  
	AU17T=$((AU17T+AU17))
	AU26T=$((AU26T+AU26)) 
	ForwardT=$((ForwardT+Forward))  
	BackwardT=$((BackwardT+Backward))  
	TotalLabsT=$((TotalLabsT + TotalLabs))
	TotalDirsT=$((TotalDirsT + 1))
	echo -e "Sujeto $label"
	echo -e "-----------------"
	echo -e "-- Smile:\t$Smile"
	echo -e "-- AU04:\t$AU04"
	echo -e "-- AU02:\t$AU02"
	echo -e "-- AU15:\t$AU15"
	echo -e "-- AU18:\t$AU18"
	echo -e "-- AU09:\t$AU09"
	echo -e "-- negAU12:\t$negAU12"
	echo -e "-- AU10:\t$AU10"
	echo -e "-- Expressive:\t$Expressive"
	echo -e "-- Unilateral_LAU12:\t$Unilateral_LAU12"
	echo -e "-- Unilateral_RAU12:\t$Unilateral_RAU12"
	echo -e "-- Unilateral_LAU14:\t$Unilateral_LAU14"
	echo -e "-- Unilateral_RAU14:\t$Unilateral_RAU14"
	echo -e "-- AU14:\t$AU14"
	echo -e "-- AU05:\t$AU05"
	echo -e "-- AU17:\t$AU17"
	echo -e "-- AU26:\t$AU26"
	echo -e "-- Forward:\t$Forward"
	echo -e "-- Backward:\t$Backward"
	echo -e "-- totalLabs:\t$TotalLabs"


done
	echo -e "TOTALES"
	echo -e "-----------------"
	echo -e "-- Smile:\t$SmileT"
	echo -e "-- AU04:\t$AU04T"
	echo -e "-- AU02:\t$AU02T"
	echo -e "-- AU15:\t$AU15T"
	echo -e "-- AU18:\t$AU18T"
	echo -e "-- AU09:\t$AU09T"
	echo -e "-- negAU12:\t$negAU12T"
	echo -e "-- AU10:\t$AU10T"
	echo -e "-- Expressive:\t$ExpressiveT"
	echo -e "-- Unilateral_LAU12:\t$Unilateral_LAU12T"
	echo -e "-- Unilateral_RAU12:\t$Unilateral_RAU12T"
	echo -e "-- Unilateral_LAU14:\t$Unilateral_LAU14T"
	echo -e "-- Unilateral_RAU14:\t$Unilateral_RAU14T"
	echo -e "-- AU14:\t$AU14T"
	echo -e "-- AU05:\t$AU05T"
	echo -e "-- AU17:\t$AU17T"
	echo -e "-- AU26:\t$AU26T"
	echo -e "-- Forward:\t$ForwardT"
	echo -e "-- Backward:\t$BackwardT"
	echo -e "-- TotalLabs:\t$TotalLabsT"
	echo -e "-- totalDirs:\t$TotalDirsT"

