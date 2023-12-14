# GraphGPS_Edge_Regularization

Check out the multiple_experiements.sh to test out experiments. 

Example: 
python main.py --cfg configs/GPS/peptides-func-GPS.yaml  --repeat 1  seed 42  wandb.use True reg '0.0' loss L1

Two parameters:
1. reg: float -- for the hyperparameter (eta) of your loss function. $\textbf{Negative Values for reg means allow full backpropagation for our regularization term. Otherwise, the regularization gradient only flows to query and key matrix transformation).}$
2. loss: {"L1", "CE"} -- means which loss function you want to run for the regularization term. 
