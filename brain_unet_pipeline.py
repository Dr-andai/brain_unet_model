# """"
# Purpose of this file
# - Automates the deployment process
# - Adds MLOps capabilities (orchestration with ZenML, tracking with MLflow).
# - Prepares project for production by managing the model lifecycle (deploy, infer, monitor).
# """
# from zenml import pipeline, step
# from zenml.integrations.huggingface.model_deployers import HuggingFaceModelDeployer
# from zenml.integrations.huggingface.services import HuggingFaceDeploymentService, HuggingFaceServiceConfig
# import mlflow
# import numpy as np
# import time
# from typing import Annotated

# @step(experiment_tracker="mlflow_tracker", enable_cache=False)
# def deploy_model() -> HuggingFaceDeploymentService:
#     """Deploy UNet model to Hugging Face Inference Endpoint."""
#     model_deployer = HuggingFaceModelDeployer.get_active_model_deployer()
#     service_config = HuggingFaceServiceConfig(
#         model_name = "brain_unet_endpoint",
#         repository = "AndaiMD/brain-unet-model",
#         task = "image-segmentation",
#         framework = "pytorch",
#         accelerator = "gpu",
#         instance_type = "nvidia-a10g",
#         instance_size = "large",
#         region = "us-east-1",
#         vendor = "azure"
#     )
#     mlflow.log_param("model_name", "AndaiMD/brain-unet-model")
#     mlflow.log_param("task", "image-segmentation")
#     # Explicitly pass service_type as the class
#     service = model_deployer.deploy_model(config=service_config, service_type=None)
#     return service

# @step(experiment_tracker="mlflow_tracker", enable_cache=False)
# def predictor(
#     service: HuggingFaceDeploymentService,
#     data: dict
# ) -> Annotated[np.ndarray, "segmentation_mask"]:
#     """Empty predictor step for testing."""
#     pass  # Returns None by default, but type hint expects np.ndarray

# @pipeline
# def brain_unet_pipeline():
#     """Pipeline to deploy and infer with UNet model."""
#     sample_data = {"image": "data/sample_mri_image.jpg"}  # Placeholder, won’t be processed
#     service = deploy_model()
#     prediction = predictor(service, sample_data)

# if __name__ == "__main__":
#     brain_unet_pipeline()