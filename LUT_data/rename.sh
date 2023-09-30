#!/bin/bash

for f in *.npy; do
    if [[ $f =~ (.*)she(.*) ]]; then
        before="${BASH_REMATCH[1]}"
        after="${BASH_REMATCH[2]}"
        new_f="${before}vcma${after}"
        mv $f $new_f
    fi
done


