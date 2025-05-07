pdfs=$(find "pdf/" -type f -iname "*.pdf")
END=200
for i in $pdfs; do
    [ -f "$i" ]     
    echo Extract text per page for "$i"
    filename=$(basename -- "$i")
    extension="${filename##*.}"
    filename="${filename%.*}" 
    for j in $(seq 1 $END); 
        do pdftotext -f $j -l $j -layout "$i" "./txt/${filename}_${j}.txt" || break; done
done