
import subprocess
import time


def start_gpulog(path, fname):
	#has to be called before start of training
	gpulogfile = open("%s/%s.csv"%(path,fname),'w')
	argument = 'timestamp,count,gpu_name,gpu_bus_id,memory.total,memory.used,utilization.gpu,utilization.memory'
	proc = subprocess.Popen(['nvidia-smi --format=csv --query-gpu=%s %s'%(argument, ' -l 1')], shell=True, stdout=gpulogfile)
	
	return proc, gpulogfile

def kill_gpulog(proc, gpulogfile):
	#has to be called after start of training
	proc.terminate()

	gpulogfile.close()
	
	
if __name__ == '__main__':
	#if we do if __name__... like here, 
	#then if this script is run directly: python gpu_stat.py
	#whats below will be executed
	#if we import this script to other file: from gpu_stat import start_gpulog
	#what is below will be not executed
	#so in this way we may do ie. testing of this script
	path = '.'
	fname = 'gpulog'
	proc, gpulogfile = start_gpulog(path, fname )
	
	time.sleep(5)
	
	kill_gpulog(proc, gpulogfile)