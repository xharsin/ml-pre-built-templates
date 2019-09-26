class MlFlowLogging:
  def __init__(self,create_log):
    self.create_log = create_log=="TRUE" # convert to boolean
    print("Create logs is set to: {}".format(self.create_log))
    self.block_runtime = timeit.default_timer() 

  def log_runtime(self,block_name):
    self.block_runtime = str(timedelta(seconds=np.round(timeit.default_timer() - self.block_runtime)))
    self.log_param('Runtime - {}'.format(block_name), self.block_runtime)
    print('{} is now finished in {}.'.format(block_name, self.block_runtime))
    self.block_runtime = timeit.default_timer()
  
  def mlflow_init(self, experiment):
    if self.create_log:
      try:
        mlflow.create_experiment(experiment)
      except:
        pass
      try:  
        mlflow.set_experiment(experiment)
        if mlflow.active_run():
          mlflow.end_run()   
        mlflow.start_run()
        print("Running with MLFlow logging: ON")
      except Exception as e:
        print("MLFlow troubles, error type",sys.exc_info()[0],str(e))
        self.create_log = False
        print("Running with MLFlow logging: OFF")
      
    
  def log_param(self, name, param):
    if self.create_log:
      try:
        mlflow.log_param(name, param)
      except Exception as e:
        print("Warning - MLFlow unavailable. Skipping logging, error type",sys.exc_info()[0],str(e))    

  def log_artifact(self, local_path, artifact_path):
    if self.create_log:
      try:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
      except Exception as e:
        print("Warning - MLFlow unavailable. Skipping logging, error type",sys.exc_info()[0],str(e))
      
  
  def log_metric(self, name, metric):
    if self.create_log:
      try:
        mlflow.log_metric(name, metric)   
      except Exception as e:
        print("Warning - MLFlow unavailable. Skipping logging, error type",sys.exc_info()[0],str(e))
 
  def log_model(self, model, path):
    if self.create_log:
      try:
        mlflow_log_model(model, path) 
      except Exception as e:
        print("Warning - MLFlow unavailable. Skipping logging, error type",sys.exc_info()[0],str(e))

  def mlflow_end(self):
    if self.create_log:
      if mlflow.active_run():
        mlflow.end_run()  