import mlrun
from kfp import dsl

    
@dsl.pipeline(
    name="LLM Feedback Loop"
)

def kfpipeline(metric_name: str, 
               input_ds):
    
    project = mlrun.get_current_project()
    
    sample = project.run_function(
        function="metric-sample",
        name="metric-sample",
        handler="sample",
        params = {"metric_name" : metric_name},
        outputs=['alert_triggered']
    )

    with dsl.Condition(sample.outputs['alert_triggered'] == "True"):

        # Generate a new DS based on the traffic
        ds = project.run_function(
            function="generate-ds",
            handler="generate_ds",
            params={"input_ds" : input_ds}, 
            outputs=["new-train-ds","dataset"])
        
        # Re-train the new model        
        train = project.run_function(
            function="train",
            params={
                "dataset": "mlrun/banking-orpo-opt",
                "base_model": "google/gemma-2b",
                "new_model": "mlrun/gemma-2b-bank-v0.2",
                "device": "cuda:0"},
            handler="train",
            outputs=["model"],
            ).after(ds)
        
        # Deploy the function with the new (re-trained) model
        deploy = project.get_function('llm-server')
        deploy.add_model(
            "google-gemma-2b",
            class_name="LLMModelServer",
            llm_type="HuggingFace",
            model_name="google/gemma-2b",
            adapter="mlrun/gemma-2b-bank-v0.2", 
            model_path=f"store://models/{project.name}/google-gemma-2b:latest",
            generate_kwargs={
                "do_sample": True,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "max_length": 80,
            },
            device_map="cuda:0",
        )
        deploy.set_tracking()
        project.deploy_function("llm-server").after(train)
        
