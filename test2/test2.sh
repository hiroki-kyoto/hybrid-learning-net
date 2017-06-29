#!/bin/bash
./bpnn.sh
./hlnn.sh
echo "================================================" >> RESULT.LOG
date >> RESULT.LOG
echo "================================================" >> RESULT.LOG
python analyze.py >> RESULT.LOG

