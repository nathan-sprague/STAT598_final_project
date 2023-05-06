import matplotlib.pyplot as plt
import numpy as np

if True: # CIFAR10
	x = [0.67, 0.73, 0.697]
	plt.bar(["Conventional\nConvolutional\nLayers","Original\nImplementation","Re-implementation"], x)
	plt.title("Testing Accuracy on the CIFAR-10 Dataset")
	plt.ylabel("Accuracy")
	plt.show()
