rm BPNN.LOG
for i in {1..10}
do
	python main.py BPNN >> BPNN.LOG
done
