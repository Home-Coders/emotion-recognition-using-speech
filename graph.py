import matplotlib.pyplot as pl

def plot_graph():
	# creating the dataset
	data = {'CNN':60, 'SVM':83, 'RNN':66,
		'RandomForest':88}
	courses = list(data.keys())
	values = list(data.values())

	fig = pl.figure(figsize = (10, 5))
	
	# creating the bar plot
	pl.bar(courses, values, color ='blue',
		width = 0.4)
	pl.xlabel("Algorithms implemented")
	pl.ylabel("Test accuracy")
	pl.title("Accuracy Comparison")
	pl.show()
  
plot_graph()
