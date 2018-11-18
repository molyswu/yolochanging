def ipython_info():
	import sys
	ip = False
	if 'ipykernel' in sys.modules:
		ip = True  # 'notebook'
	# elif 'IPython' in sys.modules:
	#    ip = 'terminal'
	return ip


ISIPYTHON = ipython_info()
DEBUG = 1

if not ISIPYTHON:
	print('working in file: ', __file__)
	import matplotlib

	# matplotlib.use('tkagg')
	matplotlib.use('agg')

else:
	print ('script is working in ipython/jupyter')


import pandas as pd, numpy as np
import matplotlib.pyplot as plt 
	
def train_plot(path, filename, show=False):
	###########
	logdata = pd.read_csv(path+'/'+filename,header=None).values

	localfps = logdata[:,0].astype(float)
	globalfps = logdata[:,1].astype(float)
	loss = logdata[:,2].astype(float)
	step = logdata[:,3].astype(int)
	predict = logdata[:,4].astype(str)
	
	fig = plt.figure(figsize=(15, 10))
	plt.title('TASK: %s' % filename)

	ax = plt.subplot(111)
	color = list(cm.rainbow(np.linspace(0, 1, 3)))

	ax.plot(step,localfps, '-o',label='local iter [fps]',c=color[0],)
	ax.plot(step, globalfps,'-.', label='global [fps]',c=color[0],)


	ax.set_xlabel('step [-]')
	ax.set_ylabel('local, global')
	ax1.tick_params('y', colors=color[0])
	
	ax2 = ax1.twinx()
	ax2.plot(step, loss, label='loss',c=color[2], )
	ax2.set_ylabel('loss')
	ax2.tick_params('y', colors=color[2])
	
	plt.grid(True)
	plt.legend()

	
	plt.savefig(path+'/'+filename.split('.')[0]+'.plot.png')
	###########

	if show:
		plt.show()
		
if __name__ == '__main__':
	path='.'
	filename = 'test.csv'
	
	with open('test.csv','w') as testcsv:
		for i in range(100):
			testcsv.write('%f,%f,%f,%d,%s\n'%(np.cos(i),np.sin(i),np.cos(i)*np.sin(i),i,'ala'))
	
	train_plot(path,filename)
	