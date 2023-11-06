$i = dir -name  -filter *md
pandoc -N -s --toc --pdf-engine=xelatex  -o zzp-llm.pdf   metadata.yaml --template=template.tex $i 