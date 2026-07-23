Everytime we run mlflow script it will create `mlruns` folder inside parent folder relative to script. so for example if we run `MLflow.ipynb` the `mlruns` folder will creates inside `notebooks` folder.

If we need to point custom location(project root is better), we need to tell it specifically.

```
mlflow.set_tracking_uri("../mlruns")
```

Also since these experiment details are private we dont version control that `mlruns` folder. so we need to update `.gitignore`