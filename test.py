import matplotlib.pyplot as plt
import numpy as np

# def plot_graph(x = [],y1 = [],y2 = [],save = False,title = "plot",xlabel = "x-axis",ylabel = "y-axis"):
# 	if len(y1)==0 or len(x)==0:
# 		return
# 	elif len(y1) != len(x):
# 		return
# 	else:
# 		if len(y2)!=0:
# 			plt.plot(x,y1,'r')
# 			plt.plot(x,y2,'b')
# 		else:
# 			plt.plot(x,y1)
# 		plt.xlabel(xlabel)
# 		plt.ylabel(ylabel)
# 		plt.title(title)
# 		if save:
# 			plt.savefig(title + ".png")
# 		plt.show()

def plot_graph(X = [],Y = [],title = "Graph",xlabel = "x-axis",ylabel = "y-axis",same_fig=False):
	if X and Y:
		if len(X)==len(Y):
			if same_fig:
				if len(plt.get_fignums())==0:
					fig,ax1 = plt.subplots()					
					ax1.set_xlabel(xlabel)
					ax1.set_ylabel(ylabel,color = 'tab:red')
					ax1.set_title(title)
					ax1.plot(X,Y,color = 'tab:red')
					plt.savefig("hha.png")					
				else:
					fig,ax1 = plt.subplots()
					ax2 = ax1.twinx()
					ax2.set_ylabel(ylabel,color = 'tab:blue')
					ax2.plot(X,Y,color = 'tab:blue')
					plt.savefig("hha.png")
					plt.show()									
			else:				
				plt.plot(X,Y)
				plt.xlabel(xlabel)
				plt.ylabel(ylabel)
				plt.title(title)
				plt.savefig("hha.png")				
				plt.close()
		else:
			print "Arguments of plot_graph does not have the same length"
	else:
		print "Argument of plot_graph is empty"
		return


X = range(1,100)
Y1 = [x**2 for x in X]
Y2 = [x**3 for x in X]
plot_graph(X = X,Y = Y1,title = "S vs V",xlabel = "S",ylabel = "V",same_fig = True)
plot_graph(X = X,Y = Y2,ylabel = "J",same_fig = True)

