#!/bin/bash

rm results.txt

python3 evaluate_test_unofficial.py --run_id 1 --subtask "test_en" >> results.txt
python3 evaluate_test_unofficial.py --run_id 1 --subtask "test_es" >> results.txt
python3 evaluate_test_unofficial.py --run_id 1 --subtask "test_fr" >> results.txt
python3 evaluate_test_unofficial.py --run_id 1 --subtask "test_pt" >> results.txt

python3 evaluate_test_unofficial.py --run_id 2 --subtask "test_en" >> results.txt
python3 evaluate_test_unofficial.py --run_id 2 --subtask "test_es" >> results.txt
python3 evaluate_test_unofficial.py --run_id 2 --subtask "test_fr" >> results.txt
python3 evaluate_test_unofficial.py --run_id 2 --subtask "test_pt" >> results.txt

python3 evaluate_test_unofficial.py --run_id 3 --subtask "test_en" >> results.txt
python3 evaluate_test_unofficial.py --run_id 3 --subtask "test_es" >> results.txt
python3 evaluate_test_unofficial.py --run_id 3 --subtask "test_fr" >> results.txt
python3 evaluate_test_unofficial.py --run_id 3 --subtask "test_pt" >> results.txt
