# GOD-Benchmark - GOD-poj104

GOD-poj104 is the first benchmark for GOD (graph object detection) problem to train and evaluate the GOD models, and is built on the poj104 dataset [1]. It consists of about 52,000 graphs of code divided into 104 groups, and contains two classes of objects of interest: various implementation of the bubble sort algorithm, and printf statements with format string vulnerabilities. 

The GOD-Benchmark contains the following content:

- GOD-poj104.7z	
- training-dataset metadata
- test-dataset metadata
- readme 

The GOD-poj104.7z consists of about 52,000 code graphs divided into 104 groups, and each graph data contains several files:
- .ast file: describes the graphical structure as an edges list.
- .nodes file: stores the features of nodes of the graph.
- .gt fils: provides the ground-truth data if the graph contains objects of interest.
- .anchors file: provides the confidence score for each class for each anchor that contains an object of the class if the graph contains objects of interest.

[1] Lili Mou, Ge Li, Lu Zhang, Tao Wang, and Zhi Jin. 2016. Convolutional neural networks over tree structures for programming language processing. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence. 1287–1293.