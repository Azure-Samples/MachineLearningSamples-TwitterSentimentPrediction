import score
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
import pandas as pd

df = pd.DataFrame(data=[['I like beautiful seattle']], columns=['input_text_string'])
df.dtypes

score.init()
input1 = pd.DataFrame(data=[['I like beautiful seattle']], columns=['input_text_string'])
result = score.run(input1)
#print(result)

inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}
generate_schema(run_func=score.run, inputs=inputs, filepath='service_schema.json')