# Node2vec and LSTM for remaining time prediction

This repository shares the code developed to apply graph embedding to solve the remaining time prediction task.

This is a Predictive Process Monitoring (PPM) task in which the predictive models (regressors) are trained to predict the remaining time of an ongoing business process instance. Thus, the input are partial instances ("prefixes"), represented by a (partial) list of events of a business process instance.  

Graph embedding (in specific, the Node2vec algorithm) is applied to generate an embedding representation of the activities registered in the events. Therefore, the resulting activity embeddings compose the part of the events' representation referring to the activities performed in each event.

LSTM is the algorithm applied to generate the regressors, but this code includes options to also generate simpler models based on the algorithms LGBM, Random Forest, Decision Tree and Linear Regression.
