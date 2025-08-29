from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("amaan-1107/text-summarizer-tokenizer")
        self.pipe = pipeline(
            "summarization",
            model = "amaan-1107/text-summarizer-model",
            tokenizer=self.tokenizer
        )
        self.gen_kwargs = {"length_penalty":0.8, "num_beams": 8, "max_length": 128}


    
    def predict(self,text):
        print("Dialogue:")
        print(text)

        output = self.pipe(text, **self.gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output