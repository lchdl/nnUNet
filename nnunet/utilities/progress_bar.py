import sys
import time
import datetime
import os

class Timer(object):
	def __init__(self, tick_now=True):
		self.time_start = time.time()
		self.time_end = 0
		if tick_now:
			self.tick()
		self.elapse_start = self.time_start
	def tick(self):
		self.time_end = time.time()
		dt = self.time_end - self.time_start
		self.time_start = self.time_end
		return dt
	def elapsed(self):
		self.elapse_end = time.time()
		return self.elapse_end - self.elapse_start
	def now(self):
		string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		return string

def printx(msg):
    '''
    single line erasable output.
    '''
    assert isinstance(msg,str),'msg must be a string object.'
    
    columns = list(os.get_terminal_size())[0]
    outsize = columns-1

    print('\r' +' '*outsize + '\r',end='')
    print(msg[0:outsize],end='')
    sys.stdout.flush()

def minibar(msg=None,a=None,b=None,time=None,fill='=',length=20):
	if length<5: length=5
	perc=a/b
	na=int((length-2)*perc)
	if na<0: na=0
	if na>length-2: na=length-2
	head = ('%s : '%msg) if len(msg)>0 else ''
	bar = '|'+fill*na+' '*(length-2-na)+'|'+' %.2f%%' %(100.0*perc)
	time_est = '%s' % (' << '+str(int(time))+'s' if (time is not None) else '')
	time_est += '%s' % ('|'+str(int(time*(b-a)/a))+'s' if (time is not None) else '') # eta
	printx(head+bar+time_est)



