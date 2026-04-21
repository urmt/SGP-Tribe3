#!/bin/bash
INPUT_TEX="paper5.tex"
INPUT_BIB="references.bib"
OUTPUT_TEX="paper5_clean.tex"
OUTPUT_BIB="references_clean.bib"
LOG_FILE="paper5_cleanup_log.txt"

echo "[START] Paper 5 Cleanup — Remove Paper 4" > $LOG_FILE
date >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

sed 's/paper3, paper4/paper3/g' $INPUT_TEX > temp_step1.tex
echo "[EDIT] Removed 'paper4' from citation blocks" >> $LOG_FILE

sed 's/Null-model subtraction provides a method for isolating the system-specific component by removing predictions derived from minimal statistical assumptions \\citep{paper3}\./Null-model subtraction provides a method for isolating the system-specific component by removing predictions derived from minimal statistical assumptions grounded in random matrix theory and intrinsic dimensionality frameworks \\citep{paper3, marchenko1967distribution, couillet2011random}./g' temp_step1.tex > temp_step2.tex
echo "[EDIT] Rewrote null-model justification" >> $LOG_FILE

awk '
/@misc{paper4,/ {skip=1; next}
skip && /^}/ {skip=0; next}
!skip {print}
' $INPUT_BIB > $OUTPUT_BIB
echo "[DELETE] Removed Paper 4 from bibliography" >> $LOG_FILE

mv temp_step2.tex $OUTPUT_TEX
rm temp_step1.tex
echo "[OUTPUT] Generated: $OUTPUT_TEX, $OUTPUT_BIB" >> $LOG_FILE

CHECK_TEX=$(grep -c "paper4" $OUTPUT_TEX)
CHECK_BIB=$(grep -c "paper4" $OUTPUT_BIB)
echo "[CHECK] paper4 in TEX: $CHECK_TEX, BIB: $CHECK_BIB" >> $LOG_FILE

cat $LOG_FILE
