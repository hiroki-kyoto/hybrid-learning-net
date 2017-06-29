rm HLNN.LOG
for i in {1..30}
do
	python main.py HLNN >> HLNN.LOG
done

