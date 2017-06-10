#!/bin/sh

LOCALDIR="/home/npellejero/tesis/AMFED/src"
REMOTEDIR="/Root/tesis/AMFED/src/"
LOG="/home/npellejero/logs/mega.log"
hostname=`labdcc`

MEGACOPY='/home/npellejero/instalaciones/megatools-1.9.96/build/bin/megacopy'
MEGARM='/home/npellejero/instalaciones/megatools-1.9.96/build/bin/megarm'

BACKUP_TIME=`date +%c`

#Obtain the files that not exists in the local server

DELETE=`$MEGACOPY --dryrun --reload --download --local $LOCALDIR --remote $REMOTEDIR | sed 's|F '$LOCALDIR'|'$REMOTEDIR'|g'`

# And remove it

for i in $DELETE; do
$MEGARM $i
done

# Run the synchronization to Mega

SYNC=`$MEGACOPY --no-progress --local $LOCALDIR --remote $REMOTEDIR`

echo "[$BACKUP_TIME][$(hostname)] synchronization to mega done!!" > $LOG
echo "Files removed $DELETE" >> $LOG
echo "Files synchronized" >> $LOG
