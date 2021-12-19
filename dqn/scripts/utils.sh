#!/bin/bash

work_should_continue() {
   if test -e $state_file; then
       return 0 
   fi
   return 1 
}

function show_time () {
    local num=$(($(date +%s)-${start_time}))
    local min=0
    local hour=0
    local day=0
    if((num>59));then
        ((sec=num%60))
        ((num=num/60))
        if((num>59));then
            ((min=num%60))
            ((num=num/60))
            if((num>23));then
                ((hour=num%24))
                ((day=num/24))
            else
                ((hour=num))
            fi
        else
            ((min=num))
        fi
    else
        ((sec=num))
    fi
    echo "Execution: $day"d "$hour"h "$min"m "$sec"s
}
