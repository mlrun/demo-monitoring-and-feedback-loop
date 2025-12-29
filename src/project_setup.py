# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import mlrun


def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project: The project to set up.

    :returns: A fully prepared project for this demo.
    """

    # Adding secrets to the projects:
    try:
        project.set_secrets(
            {
                "OPENAI_API_KEY": mlrun.get_secret_or_env("OPENAI_API_KEY"),
                "OPENAI_API_BASE": mlrun.get_secret_or_env("OPENAI_BASE_URL"),
                "HF_TOKEN": mlrun.get_secret_or_env("HF_TOKEN"),
            }
        )
    except:
        print(
            "Failed to set secrets. Please make sure you have the secrets set in the MLRun UI.")
        pass

    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image")
    image = project.get_param(key="image", default=None)
    node_selector = project.get_param(key="node_selector", default=None)
    # gpus = project.get_param(key="gpus", default=0)
    node_name = project.get_param(key="node_name", default=None)

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set or build the default image:
    if default_image is None:
        print("Building default image for the demo:")
        _build_image(project=project, image=image)
    else:
        project.set_default_image(default_image)
        project.save()

    # Set functions
    _set_function(
        project=project,
        func="model_server.py",
        name="llm-server",
        kind="serving",
        node_selector=node_selector,
        gpus=1,
        requirements=[
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "sentencepiece>=0.1.99",
        ],

    )
    _set_function(
        project=project,
        image=project.default_image,
        func="train.py",
        name="train",
        kind="job",
        gpus=1,
        node_selector=node_selector,
        node_name=node_name,
        requirements=[
            'transformers==4.56.1',
            'peft==0.17.1',
        ]
    )

    _set_function(
        project=project,
        func="generate_ds.py",
        name="generate-ds",
        kind="job",
        node_selector=node_selector,
        node_name=node_name,
        requirements=["openai==1.77.0", "huggingface-hub==0.31.1"],
        image="mlrun/mlrun",
    )

    # Save and return the project:
    project.save()
    return project


def _build_image(project: mlrun.projects.MlrunProject, image: str):
    requirements = ['tqdm==4.67.1',
                    'peft==0.17.1',
                    'trl==0.11.4',
                    'openai==1.108.0',
                    'transformers==4.56.1',
                    'datasets>=4.1.1',
                    'sentencepiece==0.2.0',
                    'deepeval==2.5.5',
                    'pyarrow==17.0.0',
                    'pydantic>=2.0',
                    'langchain==0.2.17']

    if sys.version_info.major == 3 and sys.version_info.minor == 9:
        requirements += ["protobuf==3.20.3"]

    commands = [
        # Update apt-get to install ffmpeg (support audio file formats):
        "apt-get update -y",
        # Install demo requirements:
        "pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "echo '' > /empty/requirements.txt"
    ]

    assert project.build_image(
        image=image,
        base_image="mlrun/mlrun-gpu",
        requirements=requirements,
        commands=commands,
        set_as_default=True,
    )
    project.spec.params["default_image"] = image
    project.save()


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
    gpus: int = 0,
    node_name: str = None,
    image: str = None,
    node_selector: dict = None,
    requirements: list = None,
):
    # Set the given function:
    mlrun_function = project.set_function(
        func=func,
        name=name,
        kind=kind,
        with_repo=False,
        image=image,
        requirements=requirements,
    ).apply(mlrun.auto_mount())

    # Configure GPUs according to the given kind:
    if gpus >= 1:
        mlrun_function.with_node_selection(
            node_selector=node_selector
        )
        # All GPUs for the single job:
        mlrun_function.with_limits(gpus=gpus)

        mlrun_function.spec.min_replicas = 1
        mlrun_function.spec.max_replicas = 1

    # Set the node selection:
    elif node_name:
        mlrun_function.with_node_selection(node_name=node_name)
    # Save:
    mlrun_function.save()
