#!/bin/bash
INPUT_TEX="paper5_clean.tex"
INPUT_BIB="references_clean.bib"
OUTPUT_TEX="paper5_submission.tex"
OUTPUT_BIB="references_submission.bib"
LOG_FILE="paper5_submission_log.txt"

echo "[START] Paper 5 Submission Formatter" > $LOG_FILE
date >> $LOG_FILE

# Step 1: Update documentclass
sed 's/\documentclass\[11pt\]{article}/\documentclass[preprint,12pt]{elsarticle}/g' $INPUT_TEX > temp_step1.tex
echo "[EDIT] Updated documentclass to elsarticle" >> $LOG_FILE

# Step 2: Add packages
sed '/\usepackage{caption}/a \
\usepackage{lineno} \
\journal{Neuroscience}' temp_step1.tex > temp_step2.tex
echo "[EDIT] Added lineno + journal" >> $LOG_FILE

# Step 3: Enable line numbers
sed '/\begin{document}/a \
\linenumbers' temp_step2.tex > temp_step3.tex
echo "[EDIT] Enabled line numbering" >> $LOG_FILE

# Step 4: Author block
sed 's/Mark Rowe Traver\\[0.5em\]/Mark Rowe Traver/' temp_step3.tex > temp_step4.tex
echo "[EDIT] Fixed author block" >> $LOG_FILE

# Step 5: Add keywords
sed '/\end{abstract}/a \
\begin{keyword} \
intrinsic dimensionality, multiscale analysis, neural manifolds, random matrix theory, representation geometry \
\end{keyword}' temp_step4.tex > temp_step5.tex
echo "[EDIT] Added keywords" >> $LOG_FILE

# Step 6: Figure captions
sed 's/\caption{/\\caption[T]{/g' temp_step5.tex > temp_step6.tex
echo "[EDIT] Standardized captions" >> $LOG_FILE

# Step 7: Replace [H]
sed 's/\[H\]/[htbp]/g' temp_step6.tex > temp_step7.tex
echo "[EDIT] Replaced [H] floats" >> $LOG_FILE

# Step 8: BIB style
sed 's/plainnat/elsarticle-num/g' temp_step7.tex > temp_step8.tex
cp $INPUT_BIB $OUTPUT_BIB
echo "[EDIT] Updated bib style" >> $LOG_FILE

# Step 9: Remove float package
sed '/\usepackage{float}/d' temp_step8.tex > temp_step9.tex
echo "[CLEAN] Removed float package" >> $LOG_FILE

mv temp_step9.tex $OUTPUT_TEX

LINES=$(wc -l < $OUTPUT_TEX)
CITATIONS=$(grep -c "\cite" $OUTPUT_TEX)
echo "[CHECK] Lines: $LINES, Citations: $CITATIONS" >> $LOG_FILE

cat $LOG_FILE
