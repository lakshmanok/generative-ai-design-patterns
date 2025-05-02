from unittest.mock import patch, Mock
import json

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

import lg_weather_agent

def write_mock_data(filename="chicago_weather.json"):
    data = retrieve_weather_data(41.8781, -87.6298)
    with open(filename, "w") as ofp:
        json.dump(data, ofp)
    print(f"Wrote {filename}")

# write_mock_data()

# read the hardcoded data, and use it as the return value for the function under test
with open("chicago_weather.json", "r") as ifp:
    chicago_weather = json.load(ifp)
   
@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def mock_retrieve_weather_data():
    data = lg_weather_agent.retrieve_weather_data(41.8781, -87.6298)
    print(data)

# mock_retrieve_weather_data()

app = lg_weather_agent.create_app()

@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def run_query_with_mock():
    result = lg_weather_agent.run_query(app, "Is it raining in Chicago?")
    print(result[-1])

#run_query_with_mock()

# Set up correctness metric
llm_as_judge = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model='gpt-3.5-turbo'
)


@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def eval_query_rain_today():
    input_query = "Is it raining in Chicago?"
    expected_output = "No, it is not raining in Chicago right now."
    result = lg_weather_agent.run_query(app, input_query)
    actual_output = result[-1]
    
    # metric = ContextualRecallMetric(threshold=0.5, model='gpt-3.5-turbo', include_reason=False)
    print(f"Actual: {actual_output}   Expected: {expected_output}")
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output
    )

    llm_as_judge.measure(test_case)
    print(llm_as_judge.score)

# eval_query_rain_today()


@patch('lg_weather_agent.retrieve_weather_data', Mock(return_value=chicago_weather))
def eval_dataset():
    # normally, you'll read this from a file
    dataset = [
        {
            "input_query": "Is it raining in Chicago?",
            "expected_output": "No, it is not raining in Chicago right now."
        },
        {
            "input_query": "How cold is it in Chicago?",
            "expected_output": "It's around 45F, but winds of 15-25mph makes it feel colder."
        },        
    ]
    
    for data in dataset:
        result = lg_weather_agent.run_query(app, data['input_query'])
        actual_output = result[-1]

        print(f"Actual: {actual_output}   Expected: {data['expected_output']}")
        test_case = LLMTestCase(
            input=data['input_query'],
            actual_output=actual_output,
            expected_output=data['expected_output']
        )
        
        llm_as_judge.measure(test_case)
    
    print(llm_as_judge.score)
    
eval_dataset()