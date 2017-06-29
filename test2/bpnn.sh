rm BPNN.LOG
for i in {1..30}
do
	python main.py BPNN >> BPNN.LOG
done
