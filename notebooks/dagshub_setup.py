import dagshub
import mlflow

dagshub.init(repo_owner='Anshul-0807', repo_name='mlops-mini-project', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Anshul-0807/mlops-mini-project.mlflow")
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)