"""
python -m cProfile -o profile.txt myscript.py
python -m cProfile -o profile.txt bin/timing.py -r 1 -a gradient -c normal_dist_3D_200pts_0.csv -d 0.2
"""
import argparse
import pstats
parser = argparse.ArgumentParser()
parser.add_argument("statsfile", default='profile.txt')
args = parser.parse_args()
p = pstats.Stats(args.statsfile)
p.sort_stats('cumulative').print_stats(50)
