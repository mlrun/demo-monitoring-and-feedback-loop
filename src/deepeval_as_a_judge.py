# Copyright 2024 Iguazio
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
#

import pandas as pd
import os
from typing import Any, Union
import re

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import GEval

import mlrun
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)
from mlrun.utils import logger

STATUS_RESULT_MAPPING = {
    0: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.detected,
    1: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.no_detection,
}

class DeepEvalAsAJudgeApplication(ModelMonitoringApplicationBase):
    def __init__(
        self,
        **kwargs,
    ):
        self.name = "deepeval-as-a-judge"
        self.metric_name = kwargs.pop("metric_name")
        os.environ["OPENAI_API_KEY"] = mlrun.get_secret_or_env("OPENAI_API_KEY")
        os.environ["OPENAI_BASE_URL"] = mlrun.get_secret_or_env("OPENAI_API_BASE")

    def judge(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame(columns=["question", "answer", "score", "explanation"])

        correctness_metric = GEval(
            name="Correctness",
            criteria="Correctness - determine if the actual output is related to banking.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
        )

        for i in range(len(sample_df)):
            question, answer = sample_df.loc[i, "question"], sample_df.loc[i, "answer"]
            test_case = LLMTestCase(
                input=question,
                actual_output=answer
            )
            correctness_metric.measure(test_case)

            correctness_metric.score, correctness_metric.reason

            result_df.loc[i] = [
                question,
                answer,
                correctness_metric.score,
                correctness_metric.reason,
            ]

        return result_df

    def do_tracking(
        self,
        monitoring_context,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        judge_result = self.judge(monitoring_context.sample_df)

        # log artifact:
        pattern = re.compile("[ :+.]")
        tag = re.sub(pattern, "-", str(monitoring_context.end_infer_time))
        monitoring_context.log_dataset(
            key=self.metric_name,
            df=judge_result,
            tag=tag,
        )

        # calculate value:
        mean_score = judge_result["score"].mean()

        # get status:
        status = STATUS_RESULT_MAPPING[round(mean_score)]

        return ModelMonitoringApplicationResult(
            name=self.metric_name,
            value=mean_score,
            kind=mlrun.common.schemas.model_monitoring.constants.ResultKindApp.model_performance,
            status=status,
            extra_data={},
        )