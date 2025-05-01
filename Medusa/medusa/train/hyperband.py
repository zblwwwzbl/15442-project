import numpy as np

from random import random
from math import log, ceil
from time import time, ctime
from itertools import product
import wandb
import os

os.environ["WANDB_PROJECT"] = "HyperbandMedusa"

START_SAMPLES = 200

class Hyperband:
	
  def __init__( self, get_params_function, try_params_function ):
    self.get_params = get_params_function
    self.try_params = try_params_function
		
    self.max_iter = 27 	# maximum iterations per configuration
    self.eta = 3			# defines configuration downsampling rate (default = 3)
    self.logeta = lambda x: log( x ) / log( self.eta )
    self.s_max = int( self.logeta( self.max_iter ))
    self.B = ( self.s_max + 1 ) * self.max_iter

    self.results = []	# list of dicts
    self.counter = 0
    self.best_loss = np.inf
    self.best_counter = -1
		

	# can be called multiple times
  def run( self, skip_last = 0, dry_run = False ):

    for s in reversed( range( self.s_max + 1 )):
			
      # initial number of configurations
      n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	

      # initial number of iterations per config
      r = self.max_iter * self.eta ** ( -s )		

      medusa_num_heads = [2, 3, 4, 5]
      medusa_num_layers = [1, 2]

      T = [
        {'medusa_num_heads': h, 'medusa_num_layers': l}
        for h, l in product(medusa_num_heads, medusa_num_layers)
      ]

      for i in range(( s + 1 ) - int( skip_last )):	# changed from s + 1

        # Run each of the n configs for <iterations> 
        # and keep best (n_configs / eta) configurations

        n_configs = n * self.eta ** ( -i )
        n_iterations = r * self.eta ** ( i )
        n_epochs = START_SAMPLES * n_iterations / 68623

        print(f"\n*** {n_configs} configurations x {n_epochs} iterations each")

        val_losses = []
        early_stops = []

        for t in T:
          
          self.counter += 1

          start_time = time()
          
          if dry_run:
            result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
          else:
            result = self.try_params( n_epochs, t )		# <---
            
          assert( type( result ) == dict )
          assert( 'loss' in result )
          
          seconds = int( round( time() - start_time ))
          
          loss = result['loss']	
          val_losses.append( loss )
          
          early_stop = result.get( 'early_stop', False )
          early_stops.append( early_stop )
          
          # keeping track of the best result so far (for display only)
          # could do it be checking results each time, but hey
          if loss < self.best_loss:
            self.best_loss = loss
            self.best_counter = self.counter
          
          result['counter'] = self.counter
          result['seconds'] = seconds
          result['params'] = t
          result['iterations'] = n_iterations
          # ["s", "r", "samples", "heads", "layers", "throughput"]
          name_str = "TableLog"# + str(t) + "," + str((s, i))
          self.text_table = wandb.Table(columns=["s", "iteration", "approx samples", "heads", "layers", "time per token"])
          self.text_table.add_data(s, i, n_epochs * 68623 / 2, t['medusa_num_heads'], t['medusa_num_layers'], loss)
          wandb_run = wandb.init(project="HyperbandMedusa", name=name_str, reinit=True)
          new_table = wandb.Table(columns=self.text_table.columns, data=self.text_table.data)
          wandb_run.log({"Ablations Table": new_table})

          self.results.append( result )

        # select a number of best configurations for the next loop
        # filter out early stops, if any
        indices = np.argsort( val_losses )
        T = [ T[i] for i in indices if not early_stops[i]]
        T = T[ 0:int( n_configs / self.eta )]

    return self.results