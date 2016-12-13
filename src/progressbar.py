import sys
import os
import math

bar_width = 40

def init():
	global bar_width
	bar_width = os.get_terminal_size().columns - 7
	update(0)

def update(progress):
	sys.stdout.write('\033[F') # go to beginning of previous line

	fill_count = math.ceil(progress * bar_width)
	sys.stdout.write('[' + '-' * fill_count)

	blank_count = math.floor((1.0 - progress) * bar_width)
	progress = math.ceil(progress * 100)
	padding = '   '[:-len(str(progress))]
	sys.stdout.write(' ' * blank_count + '] ' + padding + str(progress) + '%\n')
	sys.stdout.flush()